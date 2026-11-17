"""Client tests against the real mock server (ground-truth timing)."""

from __future__ import annotations

import pytest

from llmping.client import LLMClient, RequestError
from llmping.scenario import READ_FILE_TOOL
from tests.mockserver import MockOpenAIServer


async def test_chat_non_stream(mock_server):
    async with LLMClient(mock_server.base_url, "tok") as c:
        res = await c.chat("m", [{"role": "user", "content": "Reply with the single word: pong."}])
        assert "pong" in res.content
        assert res.e2e_ms is not None and res.e2e_ms > 0
        assert res.ttft_ms is None  # no first token without streaming
        assert res.usage_source == "server"


async def test_chat_stream_ground_truth_timing(mock_server):
    async with LLMClient(mock_server.base_url, "tok") as c:
        res = await c.chat(
            "m",
            [{"role": "user", "content": "Reply with the single word: pong."}],
            stream=True,
            max_tokens=8,
        )
        assert res.streamed
        assert res.content == "pong"
        # mock emits one chunk per token at 20 ms; first chunk after 20 ms.
        assert res.ttft_ms is not None and 5 <= res.ttft_ms <= 500
        assert res.e2e_ms is not None and res.e2e_ms >= res.ttft_ms


async def test_chat_stream_usage(mock_server):
    async with LLMClient(mock_server.base_url, "tok") as c:
        res = await c.chat(
            "m",
            [{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert res.usage is not None
        assert res.usage_source == "server"
        assert res.usage.completion_tokens is not None
        assert res.finish_reason in ("stop", "length")


async def test_chat_tool_calls(mock_server):
    async with LLMClient(mock_server.base_url) as c:
        res = await c.chat(
            "m",
            [{"role": "user", "content": "What is the current temperature in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_temperature",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "get_current_temperature"}},
        )
        assert res.tool_calls
        assert res.tool_calls[0].name == "get_current_temperature"
        assert res.tool_calls[0].arguments


async def test_chat_stream_captures_reasoning_content(mock_server):
    # reasoning models emit reasoning_content deltas; ping's `chars` counts
    # them, so the client must surface them. The mock only emits reasoning
    # for apples/how-many prompts.
    async with LLMClient(mock_server.base_url, "tok") as c:
        res = await c.chat(
            "m",
            [{"role": "user", "content": "How many apples do I have left?"}],
            stream=True,
            max_tokens=8,
        )
    assert res.reasoning_content  # non-empty reasoning stream captured
    assert res.content == "1"
    assert len(res.content) + len(res.reasoning_content) > len(res.content)


async def test_chat_stream_thinking_only_no_content(mock_server_think_only):
    # qwen-3.8-27b failure mode: the server streams reasoning_content but
    # zero content, then finish_reason "stop". The client must surface the
    # reasoning while leaving content empty — the scenario ok-gate keys off
    # that distinction.
    async with LLMClient(mock_server_think_only.base_url, "tok") as c:
        res = await c.chat(
            "qwen-thinker",
            [{"role": "user", "content": "Review these files for me."}],
            stream=True,
        )
    assert res.reasoning_content  # thinking happened...
    assert res.content == ""  # ...but no visible answer was emitted


async def test_chat_stream_empty_responder(mock_server_empty):
    # Some servers answer HTTP 200 with a completely empty stream (thinking
    # starved of budget): no content AND no reasoning, no first token. The
    # client must surface that signature so ping can count it as a loss.
    async with LLMClient(mock_server_empty.base_url, "tok") as c:
        res = await c.chat(
            "empty-model",
            [{"role": "user", "content": "Reply with pong."}],
            stream=True,
        )
    assert res.content == ""
    assert res.reasoning_content == ""
    assert res.ttft_ms is None  # no first content/reasoning/tool delta arrived
    # the mock honours include_usage, so tokens come from the usage block
    assert res.output_tokens is not None


async def test_output_tokens_none_without_usage(mock_server, monkeypatch):
    # A server that sends NO usage block reports no token count — a chunk
    # count is not a token count, and the fallback mislabeled frames as
    # tokens. output_tokens must be None instead.
    async def no_usage(payload):
        return {}

    async with LLMClient(mock_server.base_url, "tok") as c:
        orig = c._chat_stream

        async def strip_usage(payload):
            res = await orig(payload)
            res.usage = None
            res.usage_source = "none"
            return res

        monkeypatch.setattr(c, "_chat_stream", strip_usage)
        res = await c.chat(
            "m", [{"role": "user", "content": "Reply with the single word: pong."}], stream=True
        )
    assert res.usage is None
    assert res.streamed
    assert res.content  # frames arrived...
    assert res.output_tokens is None  # ...but no fake token count


async def test_chat_stream_parallel_tool_calls(mock_server):
    # read_file is forced → the mock emits TWO parallel calls (fib.py +
    # README.md), mirroring the real gemma-4-31b server behaviour that broke
    # the agentic session before the multi-call fix.
    async with LLMClient(mock_server.base_url) as c:
        res = await c.chat(
            "m",
            [{"role": "user", "content": "Read both files for me."}],
            stream=True,
            tools=[READ_FILE_TOOL.schema()],
            tool_choice={"type": "function", "function": {"name": "read_file"}},
        )
    assert [tc.name for tc in res.tool_calls] == ["read_file", "read_file"]
    assert "src/fib.py" in res.tool_calls[0].arguments
    assert "README.md" in res.tool_calls[1].arguments


async def test_error_status(mock_server_failing):
    async with LLMClient(mock_server_failing.base_url) as c:
        with pytest.raises(RequestError) as exc_info:
            await c.chat("fail-model", [{"role": "user", "content": "hi"}])
        assert exc_info.value.status == 503


async def test_models_list(mock_server):
    # GET {base_url}/models → the mock's advertised model IDs, in order.
    async with LLMClient(mock_server.base_url, "tok") as c:
        names = await c.models()
    assert names == MockOpenAIServer.DEFAULT_MODELS


async def test_models_list_error():
    # Unreachable endpoint → RequestError, like any other failed request.
    async with LLMClient("http://127.0.0.1:1/v1", timeout=1.0) as c:
        with pytest.raises(RequestError):
            await c.models()


async def test_connection_error():
    async with LLMClient("http://127.0.0.1:1/v1", timeout=1.0) as c:
        with pytest.raises(RequestError):
            await c.chat("m", [{"role": "user", "content": "hi"}], stream=True)
