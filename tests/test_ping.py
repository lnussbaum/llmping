"""Ping tests: loss counting, exit semantics, cadence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from llmping.client import ChatResult, LLMClient, ToolCall
from llmping.config import Target
from llmping.ping import _run_scenario, run_ping
from llmping.scenario import BUGGY_HELLO, FIXED_HELLO, HELLO_PATH, load_scenario


def _cfg(model: str = "m", scenario: str = "pingpong") -> list[Target]:
    return [Target(model=model, scenario=scenario)]


async def test_extra_body_stream_false_takes_plain_path(mock_server):
    # `--extra-body '{"stream": false}'` is a legitimate override; the
    # client must follow the *payload* (plain JSON) rather than parse a
    # non-SSE body as SSE — which reported a healthy server as an
    # empty-reply loss.
    async with LLMClient(mock_server.base_url) as client:
        res = await client.chat(
            "m",
            [{"role": "user", "content": "Reply with the single word: pong."}],
            stream=True,
            extra_body={"stream": False},
        )
    assert res.content == "pong"
    assert res.streamed is False
    assert res.usage_source == "server"  # plain path parses the usage block


async def test_extra_body_stream_false_end_to_end(mock_server):
    # and through run_ping: the round is ok, not an empty-response loss
    from llmping.ping import run_ping

    async with LLMClient(mock_server.base_url) as client:
        stats, _ = await run_ping(
            client,
            [Target(model="m", extra_body={"stream": False})],
            count=1,
            interval=0,
        )
    assert stats[0].ok, stats[0].samples[0].error
    assert stats[0].samples[0].chars == 4  # "pong"


async def test_ping_all_ok(mock_server):
    cfg = _cfg()
    async with LLMClient(mock_server.base_url) as client:
        stats, interrupted = await run_ping(client, cfg, count=2, interval=0)
    assert not interrupted
    st = stats[0].stats()
    assert st["sent"] == 2
    assert st["replies"] == 2
    assert st["loss_pct"] == 0.0
    assert st["e2e"]["min"] is not None


async def test_ping_probe_max_tokens_default_4096(mock_server, monkeypatch):
    # The probe's max_tokens is a cap, not a spend — 4096 default gives
    # thinking models headroom to answer (they'd burn a tiny budget on
    # reasoning and emit nothing).
    cfg = _cfg()
    captured: list[int] = []
    async with LLMClient(mock_server.base_url) as client:
        orig = client.chat

        async def chat_wrapper(model, messages, **kw):  # type: ignore[no-untyped-def]
            captured.append(kw.get("max_tokens"))
            return await orig(model, messages, **kw)

        monkeypatch.setattr(client, "chat", chat_wrapper)
        await run_ping(client, cfg, count=1, interval=0)
    assert captured == [4096]


async def test_ping_pingpong_is_a_real_scenario():
    # The plain probe is the default scenario: a registry entry with no
    # tools, and load_scenario resolves it like any other.
    sc = load_scenario("pingpong")
    assert sc.name == "pingpong"
    assert sc.tools == ()
    assert sc.required_tool is None
    assert sc.user_prompt == "Reply with the single word: pong."
    assert sc.turn_max_tokens == 4096
    assert sc.answer_checks == ()  # any non-empty reply counts


async def test_ping_scenario_tools_without_required_tool_rejected():
    # A scenario with tools must name one as required_tool (turn 1 forces it).
    from llmping.scenario import READ_FILE_TOOL, Scenario, ScenarioError

    with pytest.raises(ScenarioError, match="required_tool"):
        Scenario(
            name="bad",
            description="",
            system_prompt="",
            user_prompt="",
            tools=(READ_FILE_TOOL,),
            required_tool=None,
        )
    # and required_tool must be one of the scenario's own tools
    with pytest.raises(ScenarioError, match="not one of its tools"):
        replace(load_scenario("agentic-basic"), required_tool="compile")


async def test_ping_loss_detected(mock_server_failing):
    cfg = _cfg(model="fail-model")
    async with LLMClient(mock_server_failing.base_url, timeout=5) as client:
        stats, interrupted = await run_ping(client, cfg, count=2, interval=0)
    assert not interrupted
    st = stats[0].stats()
    assert st["sent"] == 2
    assert st["replies"] == 0
    assert st["loss_pct"] == 100.0
    assert not stats[0].ok  # any loss ⇒ exit 1 in the CLI


async def test_ping_chat_round_ttft_measured(mock_server):
    cfg = _cfg()
    async with LLMClient(mock_server.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    ttfts = stats[0].ttfts()
    assert len(ttfts) == 1
    assert 0 < ttfts[0] < 1000


async def test_ping_round_reports_chars(mock_server):
    cfg = _cfg()
    async with LLMClient(mock_server.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok
    assert sample.chars is not None
    assert sample.chars > 0  # mock replies "pong"
    assert stats[0].stats()["chars"] == sample.chars


async def test_ping_round_chars_includes_reasoning(mock_server, monkeypatch):
    # Real reasoning servers emit only reasoning_content (no `content`), as
    # seen on the qwen target; chars must count it so the length isn't 0.
    # The registry is built at import time, so patch the entry, not the constant.
    from dataclasses import replace

    from llmping.scenario import SCENARIOS

    monkeypatch.setitem(
        SCENARIOS,
        "pingpong",
        replace(
            load_scenario("pingpong"),
            user_prompt="How many apples do I have left after giving 2 of 3?",
        ),
    )
    cfg = _cfg()
    async with LLMClient(mock_server.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    reasoning = "Let me reason: you start with 3 and give away 2, leaving 1."
    assert stats[0].samples[0].chars == len("1") + len(reasoning)


async def test_ping_empty_response_is_loss(mock_server_empty):
    # A 200 stream that produced no text at all (content AND reasoning empty —
    # the qwen thinking-starved `tokens=8 chars=0 ttft=None` signature) is a
    # loss, not a reply.
    cfg = _cfg(model="empty-model")
    async with LLMClient(mock_server_empty.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok is False
    assert sample.chars == 0
    assert sample.turns == 1
    assert sample.ttft_ms is None
    assert sample.error == "empty response (0 chars)"
    assert stats[0].stats()["loss_pct"] == 100.0
    # e2e/ttft stats exclude the lost sample
    assert stats[0].stats()["e2e"]["avg"] is None


def _agentic_cfg() -> list[Target]:
    return _cfg(scenario="agentic-basic")


async def test_ping_agentic_basic_session(mock_server, monkeypatch):
    # Mini agentic session: model must call read_file (mock emits TWO
    # parallel calls — fib.py + README.md), mocked exec returns both file
    # contents, model answers on turn 2.
    cfg = _agentic_cfg()
    async with LLMClient(mock_server.base_url) as client:
        sent: list[list[dict]] = []
        orig = client.chat

        async def chat_wrapper(model, messages, **kw):  # type: ignore[no-untyped-def]
            sent.append(messages)
            return await orig(model, messages, **kw)

        monkeypatch.setattr(client, "chat", chat_wrapper)
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok, sample.error
    assert sample.turns == 2  # forced tool turn + final answer
    assert sample.ttft_ms is not None
    assert sample.chars is not None and sample.chars > 0
    assert sample.output_tokens is not None and sample.output_tokens > 0
    # regression: the turn-2 request must hand back BOTH parallel tool
    # results (the gemma-4-31b case that left the model waiting on the
    # missing file and answered nothing).
    turn2 = sent[-1]
    tool_msgs = [m for m in turn2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    blob = json.dumps(turn2)
    assert "src/fib.py" in blob and "README.md" in blob


async def test_ping_agentic_multi_turn_single_file_per_turn(mock_server_serial_reader, monkeypatch):
    # qwen scenario: the model reads ONE file per tool turn (fib.py, then
    # README.md) and only answers on turn 3. The loop must carry the
    # session through to the answer.
    cfg = _cfg(model="qwen-serial", scenario="agentic-basic")
    async with LLMClient(mock_server_serial_reader.base_url) as client:
        sent: list[list[dict]] = []
        orig = client.chat

        async def chat_wrapper(model, messages, **kw):  # type: ignore[no-untyped-def]
            sent.append(list(messages))  # snapshot: `messages` is mutated between turns
            return await orig(model, messages, **kw)

        monkeypatch.setattr(client, "chat", chat_wrapper)
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok, sample.error
    assert sample.turns == 3  # read fib.py → read README.md → answer
    assert len(sent) == 3
    # turn-2 request carries ONE tool result (only fib.py); turn-3 carries both
    assert len([m for m in sent[1] if m.get("role") == "tool"]) == 1
    assert len([m for m in sent[2] if m.get("role") == "tool"]) == 2
    blob = json.dumps(sent[2])
    assert "src/fib.py" in blob and "README.md" in blob


async def test_ping_agentic_turn_cap_is_loss(mock_server_serial_reader):
    # A model that keeps calling the tool without ever answering hits the
    # turn cap → loss, not an infinite loop. max_turns=2 cuts the session off
    # right after the second tool turn (README.md still unread → no answer).
    target = Target(model="qwen-serial", scenario="agentic-basic")
    cap_scenario = replace(load_scenario("agentic-basic"), max_turns=2)
    async with LLMClient(mock_server_serial_reader.base_url) as client:
        probe = await _run_scenario(client, target, cap_scenario)
    assert probe["ok"] is False
    assert probe["turns"] == 2  # the cap cut it off before answering
    assert "max_turns=2" in probe["error"]


async def test_ping_agentic_thinking_only_is_loss(mock_server_think_only):
    # qwen-3.8-27b failure mode: the model calls read_file (tool gate ok),
    # then on turn 2 streams ONLY reasoning_content and stops — no visible
    # final answer. Reasoning alone must NOT satisfy the ok gate.
    cfg = _cfg(model="qwen-thinker", scenario="agentic-basic")
    async with LLMClient(mock_server_think_only.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok is False
    assert sample.turns == 2  # forced tool turn + the empty thinking-only turn
    assert "no final answer" in sample.error
    # reasoning WAS streamed (chars sum it), yet the round is still a loss
    assert sample.chars is not None and sample.chars > 0


async def test_ping_agentic_wrong_answer_is_loss(mock_server_wrong_answer):
    # The model calls read_file and produces non-empty text, but it never
    # mentions the file facts (no fib, no 42) — the scenario's fuzzy answer
    # validation must reject it as a loss, not count it as a reply.
    cfg = _cfg(model="wrong-answer", scenario="agentic-basic")
    async with LLMClient(mock_server_wrong_answer.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok is False
    assert "answer missing" in sample.error
    assert "fib function" in sample.error
    assert "magic constant" in sample.error
    assert stats[0].stats()["loss_pct"] == 100.0


async def test_ping_count_inf_cancel_returns_partial_stats(mock_server):
    # Ctrl-C under `asyncio.run` (3.11+) arrives as CancelledError on the main
    # task (its SIGINT handler cancels it), not KeyboardInterrupt. run_ping
    # must absorb it and return the partial stats with interrupted=True so the
    # CLI can still print the summary (and the JSON report on stdout)
    # before exit 130.
    import asyncio

    cfg = _cfg()
    async with LLMClient(mock_server.base_url) as client:
        task = asyncio.create_task(run_ping(client, cfg, count=None, interval=0.01))
        # the mock takes ~100 ms per round; let at least one complete
        await asyncio.sleep(0.3)
        task.cancel()
        stats, interrupted = await task  # no CancelledError escapes
        # run_ping acknowledged the cancellation, so the task is not left cancelled.
        assert task.cancelling() == 0
    assert interrupted is True
    assert stats[0].stats()["sent"] >= 1  # partial results survived the cancel


# --- agentic pipeline: read → write → compile → run ---------------------------------


class _ScriptedClient:
    """A stand-in LLMClient that replays scripted ChatResults (tool calls then a
    final answer), so the scenario pipeline gates are testable without a server."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    async def chat(self, model, messages, **kw):  # type: ignore[no-untyped-def]
        self.requests.append(list(messages))  # snapshot: `messages` is mutated between turns
        return self.replies.pop(0)


def _tool_reply(name: str, args: dict) -> ChatResult:
    return ChatResult(
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(args))],
        ttft_ms=1.0,
        e2e_ms=1.0,
        streamed=True,
    )


def _answer_reply(text: str) -> ChatResult:
    return ChatResult(content=text, ttft_ms=1.0, e2e_ms=1.0, streamed=True)


async def test_ping_agentic_full_pipeline_ok():
    # An honest agent walks the whole read → write → compile → run pipeline,
    # then answers with the program output. Every ping-side gate must pass.
    client = _ScriptedClient(
        [
            _tool_reply("read_source", {"path": HELLO_PATH}),
            _tool_reply("write_source", {"path": HELLO_PATH, "content": FIXED_HELLO}),
            _tool_reply("compile", {"path": HELLO_PATH}),
            _tool_reply("run", {"path": HELLO_PATH}),
            _answer_reply("The fixed program prints Hello, world!"),
        ]
    )
    target = Target(model="m", scenario="agentic")
    probe = await _run_scenario(client, target, load_scenario("agentic"))
    assert probe["ok"] is True, probe["error"]
    assert probe["turns"] == 5  # read → write → compile → run → answer
    # every pipeline step was actually executed by the ping side
    last = client.requests[-1]
    assert len([m for m in last if m.get("role") == "tool"]) == 4


async def test_ping_agentic_skipped_steps_is_loss():
    # The model only reads (and never touches write/compile/run) then answers
    # — agentic requires the whole pipeline; reading alone is not enough.
    # required_tools must reject it.
    client = _ScriptedClient(
        [
            _tool_reply("read_source", {"path": HELLO_PATH}),
            _answer_reply("The fixed program prints Hello, world!"),
        ]
    )
    target = Target(model="m", scenario="agentic")
    probe = await _run_scenario(client, target, load_scenario("agentic"))
    assert probe["ok"] is False
    assert probe["turns"] == 2
    assert probe["error"] == "required tools not used: write_source, compile, run"


async def test_ping_agentic_compile_failure_is_loss():
    # The model "writes" the BUGGY source: the mocked compile rejects it and run
    # refuses to run. The final answer looks right (mentions the output) and
    # answer_checks pass — but required_success (compile must be ok) must still
    # make it a loss. This is what makes the pipeline gate more than cosmetic.
    client = _ScriptedClient(
        [
            _tool_reply("read_source", {"path": HELLO_PATH}),
            _tool_reply("write_source", {"path": HELLO_PATH, "content": BUGGY_HELLO}),
            _tool_reply("compile", {"path": HELLO_PATH}),
            _tool_reply("run", {"path": HELLO_PATH}),
            _answer_reply("The fixed program prints Hello, world!"),
        ]
    )
    target = Target(model="m", scenario="agentic")
    probe = await _run_scenario(client, target, load_scenario("agentic"))
    assert probe["ok"] is False
    assert "compile did not succeed" in probe["error"]
    assert "pritn" in probe["error"]  # surfaced the mocked compiler's error


async def test_ping_agentic_pipeline_session(mock_server):
    # End-to-end through the real mock server: it plays the ideal agent over
    # the whole pipeline (one step per turn), the ping side executes each
    # mocked step and the gates require all four tools + compile/run success.
    cfg = _cfg(scenario="agentic")
    async with LLMClient(mock_server.base_url) as client:
        stats, _ = await run_ping(client, cfg, count=1, interval=0)
    sample = stats[0].samples[0]
    assert sample.ok, sample.error
    assert sample.turns == 5  # read → write → compile → run → answer
    assert sample.ttft_ms is not None
    assert sample.chars is not None and sample.chars > 0
    assert stats[0].stats()["loss_pct"] == 0.0
