"""Deterministic mock OpenAI-compatible server (test-only).

A pure-ASGI app with known ground truth: chunks emitted at exact intervals,
deterministic tool-call responses, deterministic reasonings. Lives under
`tests/` because its only consumer is the test suite (see conftest); it can
still be run in-process with uvicorn as a GPU-free loopback target
(`python -m tests.mockserver.serve --port 8321`).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from llmping.scenario import FIXED_HELLO, HELLO_PATH, tool_results


def _content_for(prompt: str) -> str:
    """Deterministic response text for a prompt, so checks are stable."""
    p = prompt.lower()
    if "pong" in p:
        return "pong"
    if "fib.py" in p or "fibonacci" in p or "magic constant" in p:
        # the agentic-basic scenario's answer: must satisfy its fuzzy fact checks
        return (
            "src/fib.py defines the fib(n) Fibonacci function, "
            "and the magic constant in README.md is 42."
        )
    if HELLO_PATH in p or "hello, world!" in p:
        # the agentic scenario's answer: must satisfy its fuzzy fact checks
        return "The fixed program prints Hello, world!"
    if "count from 1 to 5" in p:
        return "1\n2\n3\n4\n5"
    if "how many do i have left" in p or "apples" in p:
        return "1"
    if "hello" in p:
        return "hello there"
    return "This is a deterministic mock response about machine learning."


def _reasoning_for(prompt: str) -> str:
    if "apples" in prompt.lower() or "how many" in prompt.lower():
        return "Let me reason: you start with 3 and give away 2, leaving 1."
    return ""


def _reply_for(prompt: str, model: str, srv: MockOpenAIServer) -> tuple[str, str]:
    """Deterministic (content, reasoning) for a plain (non-tool) reply,
    honouring the model-behaviour presets:
    - `empty_models`: nothing at all — the thinking-starved signature
      (HTTP 200, 0 content, 0 reasoning, no ttft);
    - `think_only_models`: reasoning but no visible content;
    - `wrong_answer_models`: non-empty content that fails the scenario's
      fuzzy fact checks (a generic "I read the files" reply).
    """
    if model in srv.wrong_answer_models:
        return "I read the files and everything looks fine.", ""
    content = _content_for(prompt)
    reasoning = _reasoning_for(prompt)
    if model in srv.empty_models:
        return "", ""
    if model in srv.think_only_models:
        return "", reasoning or "Let me think about the files carefully, then stop."
    return content, reasoning


def _prompt_from(messages: list) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "\n".join(parts)


def _usage(prompt_chars: int, completion_tokens: int, system_chars: int = 0) -> dict[str, Any]:
    prompt_tokens = max(1, prompt_chars // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": max(0, system_chars // 4)},
    }


MOCK_FILES = ["src/fib.py", "README.md"]


def _mock_unread(messages: list) -> list[str]:
    """Scenario files the model still needs to read, from the `tool` results
    already present in the request (the mock replays read_file deterministically
    from prior turns)."""
    read: set[str] = set()
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        try:
            path = json.loads(content).get("path")
        except (ValueError, TypeError):
            continue
        if isinstance(path, str):
            read.add(path)
    return [p for p in MOCK_FILES if p not in read]


# Deterministic "ideal agent" for the agentic pipeline: read the buggy
# source, write the FIXED version, compile, then run. Each step is one tool
# call per turn; once run has reported, the mock answers.
_FIX_HELLO_ARGS: dict[str, dict[str, Any]] = {
    "read_source": {"path": HELLO_PATH},
    "write_source": {"path": HELLO_PATH, "content": FIXED_HELLO},
    "compile": {"path": HELLO_PATH},
    "run": {"path": HELLO_PATH},
}


def _agentic_next(messages: list) -> str | None:
    """Next agentic-pipeline tool call the ideal agent would make, or None to answer."""
    used = set(tool_results(messages))
    if "read_source" not in used:
        return "read_source"
    if "write_source" not in used:
        return "write_source"
    if "compile" not in used:
        return "compile"
    if "run" not in used:
        return "run"
    return None


def _tool_calls_spec(
    call_tool: str | None, paths: list[str], tool_args: dict[str, Any] | None
) -> list[tuple[str, str, str]]:
    """(id, name, args-json) for a tool-call turn, or [] for a content turn."""
    if call_tool is None:
        return []
    if tool_args is not None:
        return [("call_mock1", call_tool, json.dumps(tool_args))]
    if call_tool == "read_file":
        return [
            (f"call_read{i + 1}", "read_file", json.dumps({"path": p})) for i, p in enumerate(paths)
        ]
    return [("call_mock1", call_tool, json.dumps({"location": "Paris, France"}))]


class MockOpenAIServer:
    """Config for the mock: chunk timing and error injection."""

    DEFAULT_MODELS = ["gpt-oss-120b", "mistral-small-4-119b"]

    def __init__(
        self,
        first_chunk_ms: float = 25.0,
        chunk_ms: float = 25.0,
        fail_models: set[str] | None = None,
        think_only_models: set[str] | None = None,
        serial_reader_models: set[str] | None = None,
        empty_models: set[str] | None = None,
        wrong_answer_models: set[str] | None = None,
        models: list[str] | None = None,
    ):
        self.first_chunk_ms = first_chunk_ms
        self.chunk_ms = chunk_ms
        self.fail_models = fail_models or set()
        self.think_only_models = think_only_models or set()
        self.serial_reader_models = serial_reader_models or set()
        self.empty_models = empty_models or set()
        self.wrong_answer_models = wrong_answer_models or set()
        # Order is significant: this is exactly what GET /models advertises.
        self.models = list(models) if models is not None else list(self.DEFAULT_MODELS)


def make_app(server: MockOpenAIServer | None = None) -> Any:
    srv = server or MockOpenAIServer()

    async def _send_error(scope, send, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _read_body(receive) -> bytes:
        body = b""
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                continue
            chunk = msg.get("body", b"")
            if chunk is not None and not isinstance(chunk, bytes):
                continue
            body += chunk or b""
            if not msg.get("more_body", False):
                return body

    async def app(scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        path = scope.get("path", "/")
        method = scope.get("method", "GET")

        if method == "GET" and path == "/v1/models":
            models_payload: dict[str, Any] = {
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "created": 0, "owned_by": "mock"}
                    for name in srv.models
                ],
            }
            bytes_ = json.dumps(models_payload).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": bytes_})
            return

        body = await _read_body(receive)

        if method == "POST" and path == "/v1/chat/completions":
            chat_data: dict[str, Any] = json.loads(body.decode())
            model = chat_data.get("model", "")
            if model in srv.fail_models:
                await _send_error(scope, send, 503, f"model {model} is loading")
                return
            messages = chat_data.get("messages", [])
            prompt = _prompt_from(messages)
            stream = bool(chat_data.get("stream"))
            include_usage = bool((chat_data.get("stream_options") or {}).get("include_usage"))
            tools = chat_data.get("tools") or []
            tool_choice = chat_data.get("tool_choice")
            forced_name = None
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                forced_name = (tool_choice.get("function") or {}).get("name")
            tool_name = forced_name or (tools[0]["function"]["name"] if tools else None)
            # Plan this exchange:
            #  - read_file: replay the still-unread mock files (from prior
            #    `tool` results). serial_reader_models read ONE file per tool
            #    turn (the qwen-3.8-27b behaviour); everyone else batches the
            #    remaining files in parallel (gemma-4-31b). Nothing left to
            #    read → plain content answer.
            #  - agentic (toolbox has write_source): play the ideal agent —
            #    one pipeline step per turn (read → write → compile → run),
            #    then a content answer. The mocked executors on the ping side
            #    already validate the written source, so the session never
            #    stalls here.
            #  - any other tool: the classic single forced call.
            call_tool = None
            paths: list[str] = []
            tool_args: dict[str, Any] | None = None
            if any((t.get("function") or {}).get("name") == "write_source" for t in tools):
                step = _agentic_next(messages)
                if step:
                    call_tool = step
                    tool_args = _FIX_HELLO_ARGS[step]
            elif tool_name == "read_file":
                paths = _mock_unread(messages)
                if paths:
                    call_tool = "read_file"
                    if model in srv.serial_reader_models:
                        paths = paths[:1]
            elif tool_name:
                call_tool = tool_name

            system_chars = sum(
                len(m.get("content", "")) for m in messages if m.get("role") == "system"
            )

            if stream:
                await _stream_chat(
                    send,
                    srv,
                    chat_data,
                    prompt,
                    call_tool,
                    paths,
                    tool_args,
                    include_usage,
                    system_chars,
                )
            else:
                content, reasoning = _reply_for(prompt, model, srv)
                message: dict[str, Any] = {"role": "assistant"}
                if reasoning:
                    message["reasoning_content"] = reasoning
                calls = _tool_calls_spec(call_tool, paths, tool_args)
                if calls:
                    message["content"] = None
                    message["tool_calls"] = [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": cname, "arguments": cargs},
                        }
                        for cid, cname, cargs in calls
                    ]
                else:
                    message["content"] = content
                chat_payload: dict[str, Any] = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": _usage(len(prompt), max(1, len(content) // 4), system_chars),
                }
                bytes_ = json.dumps(chat_payload).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": bytes_})
            return

        await _send_error(scope, send, 404, f"not found: {method} {path}")

    return app


async def _stream_chat(
    send,
    srv: MockOpenAIServer,
    data: dict,
    prompt: str,
    call_tool: str | None,
    paths: list[str],
    tool_args: dict[str, Any] | None,
    include_usage: bool,
    system_chars: int,
) -> None:
    """SSE stream with known ground-truth chunk timing."""
    headers = [(b"content-type", b"text/event-stream")]
    await send({"type": "http.response.start", "status": 200, "headers": headers})
    model = data.get("model", "")
    max_tokens = int(data.get("max_tokens", 64) or 64)
    n_chunks = min(max_tokens, 32)
    n_chunks = max(n_chunks, 1)

    async def emit(obj: dict[str, Any]) -> None:
        await send(
            {
                "type": "http.response.body",
                "body": f"data: {json.dumps(obj)}\n\n".encode(),
                "more_body": True,
            }
        )

    await asyncio.sleep(srv.first_chunk_ms / 1000.0)

    if call_tool:
        # read_file replays the planned unread files (parallel unless the
        # model is a serial reader taking one file per tool turn); agentic
        # emits one pipeline step per turn with its canned args; other
        # forced tools stay single-call.
        calls = _tool_calls_spec(call_tool, paths, tool_args)
        for idx, (cid, cname, cargs) in enumerate(calls):
            for i, piece in enumerate(_split(cargs, 4)):
                tool_delta: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": idx,
                            "id": cid if i == 0 else None,
                            "type": "function",
                            "function": {"name": cname if i == 0 else None, "arguments": piece},
                        },
                    ],
                }
                await emit(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": tool_delta, "finish_reason": None}],
                    }
                )
                await asyncio.sleep(srv.chunk_ms / 1000.0)
        await emit(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
    else:
        content, reasoning = _reply_for(prompt, model, srv)
        pieces = _split(content, max(1, len(content) // max(n_chunks, 1) or 1))
        for i, piece in enumerate(pieces):
            delta: dict[str, Any] = {"role": "assistant", "content": piece}
            if i == 0 and reasoning:
                delta["reasoning_content"] = reasoning
            await emit(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
            await asyncio.sleep(srv.chunk_ms / 1000.0)
        await emit(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )

    if include_usage:
        completion_tokens = sum(1 for _ in re.findall(r"\s", content)) if not call_tool else 8
        completion_tokens = max(completion_tokens, 1)
        await emit(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [],
                "usage": _usage(len(prompt), completion_tokens, system_chars),
            }
        )

    await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})


def _split(s: str, size: int) -> list[str]:
    if not s:
        return [""]
    return [s[i : i + size] for i in range(0, len(s), size)]
