"""Async httpx core: manual SSE parsing,
client-side timestamps, NO silent retries — a retry would corrupt latency
and loss semantics. All traffic goes over httpx (chat completions and
model listing).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from llmping.rundir import redact_headers

DEFAULT_TIMEOUT = 30.0


class RequestError(Exception):
    """A request failed (HTTP error, timeout, transport). Counts as loss."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class ToolCall:
    id: str | None
    name: str
    arguments: str


@dataclass
class Usage:
    # Only what ping consumes: the completion count. The full raw block is
    # still recorded verbatim in --dump-traffic when debugging.
    completion_tokens: int | None = None


def _parse_usage(raw: dict | None) -> Usage | None:
    if not raw:
        return None
    return Usage(completion_tokens=raw.get("completion_tokens"))


@dataclass
class ChatResult:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    streamed: bool = False
    usage_source: str = "none"  # "server" | "none"

    @property
    def output_tokens(self) -> int | None:
        """Server-reported completion tokens, or None when the server sent
        no usage block — a chunk count is NOT a token count, and a
        mislabeled number is worse than a missing one."""
        if self.usage and self.usage.completion_tokens is not None:
            return self.usage.completion_tokens
        return None


class LLMClient:
    """One async httpx client for ping. base_url is the API prefix; the
    endpoints are `/chat/completions` (ping) and `/models` (--list-models)."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        dump_exchange: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._dump = dump_exchange
        self._client: httpx.AsyncClient | None = None

    def _record_exchange(
        self,
        path: str,
        payload: dict | None,
        status: int | None,
        resp_headers: dict[str, str],
        body_raw: str | None,
        decoded: Any = None,
        *,
        method: str = "POST",
        note: str | None = None,
    ) -> None:
        """One JSONL exchange line (opt-in via --dump-traffic).

        Bodies appear twice: `body_raw` (whatever hit the wire — SSE frames
        as text) and `body` (decoded — parsed JSON dict, or a list of parsed
        frames for streams). Credential-bearing headers are masked. GETs
        (e.g. `/models`) pass `method="GET"` and no request body.
        """
        if self._dump is None:
            return
        request: dict[str, Any] = {
            "method": method,
            "url": self.base_url + path,
            "headers": {"authorization": "***"} if self.token else {},
        }
        if payload is not None:
            request["body"] = payload
        self._dump(
            {
                "kind": path.strip("/").replace("/", "."),
                "request": request,
                "response": {
                    "status": status,
                    "headers": redact_headers(resp_headers),
                    "body_raw": body_raw,
                    "body": decoded,
                },
                "note": note,
            }
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                # httpx never retries by default; explicit here for emphasis.
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _request(
        self, method: Literal["GET", "POST"], path: str, payload: dict | None = None
    ) -> httpx.Response:
        """One non-streaming request, sharing timeout/HTTP error handling
        (no retries, like the stream path). Failures are recorded."""
        http = await self._get_client()
        try:
            if method == "GET":
                resp = await http.get(path)
            else:
                resp = await http.post(path, json=payload)
        except httpx.TimeoutException as e:
            self._record_exchange(
                path,
                payload,
                None,
                {},
                None,
                method=method,
                note=f"timeout after {self.timeout}s",
            )
            raise RequestError(f"timeout after {self.timeout}s") from e
        except httpx.HTTPError as e:
            self._record_exchange(
                path,
                payload,
                None,
                {},
                None,
                method=method,
                note=f"{type(e).__name__}: {e}",
            )
            raise RequestError(f"{type(e).__name__}: {e}") from e
        if resp.status_code >= 400:
            body = resp.text[:300]
            self._record_exchange(
                path, payload, resp.status_code, dict(resp.headers), body, method=method
            )
            raise RequestError(f"HTTP {resp.status_code}: {body}", status=resp.status_code)
        return resp

    # ------------------------------------------------------------------ models
    async def models(self) -> list[str]:
        """List the model IDs the server advertises (GET {base_url}/models).

        Standard OpenAI-compatible discovery endpoint. Returns the `id` of
        every entry in `data`, in server order. Raises RequestError on
        transport/HTTP errors (no retries, like `chat`).
        """
        resp = await self._request("GET", "/models")
        data = resp.json()
        self._record_exchange(
            "/models",
            None,
            resp.status_code,
            dict(resp.headers),
            resp.text,
            decoded=data,
            method="GET",
        )
        entries = data.get("data") if isinstance(data, dict) else None
        return [e["id"] for e in (entries or []) if isinstance(e, dict) and e.get("id")]

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stop: list[str] | str | None = None,
        response_format: dict | None = None,
        extra_body: dict | None = None,
        stream_options: dict | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        if stop is not None:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)  # shallow merge into top level
        # Branch on the *payload*, not the argument: `--extra-body
        # '{"stream": false}'` is a legitimate override (e.g. a flaky
        # proxy), and the wire format must follow it — parsing a plain
        # JSON body as SSE reports a healthy server as an empty-reply loss.
        if payload["stream"]:
            if stream_options is None:
                stream_options = {"include_usage": True}
            payload["stream_options"] = stream_options
            return await self._chat_stream(payload)
        return await self._chat_plain(payload)

    async def _chat_plain(self, payload: dict) -> ChatResult:
        t0 = time.perf_counter()
        resp = await self._request("POST", "/chat/completions", payload)
        data = resp.json()
        self._record_exchange(
            "/chat/completions",
            payload,
            resp.status_code,
            dict(resp.headers),
            resp.text,
            decoded=data,
        )
        result = ChatResult(usage_source="none")
        # The plain (non-stream) path has no first-token time — ttft stays
        # None — but e2e is the whole request latency (the path is
        # reachable from ping via `--extra-body '{"stream": false}'`).
        result.e2e_ms = (time.perf_counter() - t0) * 1000
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        result.content = msg.get("content") or ""
        result.reasoning_content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        result.finish_reason = choice.get("finish_reason")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            result.tool_calls.append(
                ToolCall(
                    id=tc.get("id"), name=fn.get("name") or "", arguments=fn.get("arguments") or ""
                )
            )
        usage = _parse_usage(data.get("usage"))
        if usage:
            result.usage = usage
            result.usage_source = "server"
        return result

    async def _chat_stream(self, payload: dict) -> ChatResult:
        result = ChatResult(streamed=True)
        first_token_t: float | None = None
        last_chunk_t: float | None = None
        tool_acc: dict[int, dict[str, Any]] = {}
        t0 = time.perf_counter()
        http = await self._get_client()
        raw_lines: list[str] = []
        frames: list[Any] = []
        status = 200
        resp_headers: dict[str, str] = {}
        try:
            async with http.stream("POST", "/chat/completions", json=payload) as resp:
                status = resp.status_code
                resp_headers = dict(resp.headers)
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    self._record_exchange("/chat/completions", payload, status, resp_headers, body)
                    raise RequestError(f"HTTP {resp.status_code}: {body}", status=resp.status_code)
                async for line in resp.aiter_lines():
                    raw_lines.append(line)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        frames.append("[DONE]")
                        break
                    now = time.perf_counter()
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        frames.append(data_str)  # unparseable frame kept verbatim
                        continue
                    frames.append(chunk)
                    last_chunk_t = now

                    usage = _parse_usage(chunk.get("usage"))
                    if usage:
                        result.usage = usage
                        result.usage_source = "server"
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            result.finish_reason = choice["finish_reason"]
                        content = delta.get("content")
                        if content:
                            result.content += content
                            if first_token_t is None:
                                first_token_t = now
                        rc = delta.get("reasoning_content") or delta.get("reasoning")
                        if rc:
                            result.reasoning_content += rc
                            if first_token_t is None:
                                first_token_t = now
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            acc = tool_acc.setdefault(idx, {"id": None, "name": "", "args": []})
                            if tc.get("id"):
                                acc["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                acc["name"] = fn["name"]
                            if fn.get("arguments"):
                                acc["args"].append(fn["arguments"])
                            if first_token_t is None:
                                first_token_t = now
            self._record_exchange(
                "/chat/completions",
                payload,
                status,
                resp_headers,
                "\n".join(raw_lines),
                decoded=frames,
            )
        except httpx.TimeoutException as e:
            self._record_exchange(
                "/chat/completions",
                payload,
                status,
                resp_headers,
                "\n".join(raw_lines),
                decoded=frames,
                note=f"timeout after {self.timeout}s",
            )
            raise RequestError(f"timeout after {self.timeout}s") from e
        except httpx.HTTPError as e:
            self._record_exchange(
                "/chat/completions",
                payload,
                status,
                resp_headers,
                "\n".join(raw_lines),
                decoded=frames,
                note=f"{type(e).__name__}: {e}",
            )
            raise RequestError(f"{type(e).__name__}: {e}") from e

        end_t = last_chunk_t if last_chunk_t is not None else time.perf_counter()
        result.ttft_ms = (first_token_t - t0) * 1000 if first_token_t is not None else None
        result.e2e_ms = (end_t - t0) * 1000
        result.tool_calls = [
            ToolCall(id=a["id"], name=a["name"], arguments="".join(a["args"]))
            for _, a in sorted(tool_acc.items())
        ]
        return result
