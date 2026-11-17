"""`ping` — repeated minimal probes, ping-style live output."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from llmping import metrics
from llmping.client import LLMClient, RequestError, ToolCall
from llmping.config import Target
from llmping.scenario import (
    History,
    Scenario,
    ScenarioError,
    check_answer,
    load_scenario,
    tool_results,
)


def _len_chars(*texts: str) -> int:
    return sum(len(t) for t in texts)


class ProbeResult(TypedDict):
    """What one probe/scenario session reports (turns into a `Sample`)."""

    ok: bool
    error: str | None
    ttft_ms: float | None
    e2e_ms: float | None
    output_tokens: int | None
    chars: int | None
    turns: int | None


class ModelStats(TypedDict):
    """One model's summary block in the report (the run's single model)."""

    sent: int
    replies: int
    loss_pct: float
    e2e: metrics.LatencyStats
    ttft: metrics.LatencyStats
    tokens: int
    chars: int
    turns: int


@dataclass
class Sample:
    model: str
    seq: int
    ok: bool = True
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int | None = None
    chars: int | None = None
    turns: int | None = None
    error: str | None = None


@dataclass
class PingStats:
    model: str
    samples: list[Sample] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.samples)

    def e2es(self) -> list[float]:
        return [s.e2e_ms for s in self.samples if s.ok and s.e2e_ms is not None]

    def ttfts(self) -> list[float]:
        return [s.ttft_ms for s in self.samples if s.ok and s.ttft_ms is not None]

    def stats(self) -> ModelStats:
        return {
            "sent": len(self.samples),
            "replies": sum(1 for s in self.samples if s.ok),
            "loss_pct": round(
                100.0 * sum(1 for s in self.samples if not s.ok) / max(len(self.samples), 1), 1
            ),
            "e2e": metrics.ping_stats(self.e2es()),
            "ttft": metrics.ping_stats(self.ttfts()),
            "tokens": sum(s.output_tokens or 0 for s in self.samples),
            "chars": sum(s.chars or 0 for s in self.samples),
            "turns": sum(s.turns or 0 for s in self.samples),
        }


def _tool_continuation(
    scenario: Scenario, calls: list[ToolCall], history: History | None = None
) -> list[dict[str, Any]]:
    """assistant tool_calls message + one mocked `tool` result per call.

    Parallel calls are handled: every call the model emitted (not just the
    required one) gets its result, so the final turn has the full context.
    Each result comes from the scenario's mocked executor — no real I/O.
    Executors may inspect `history` (the conversation so far) to reason about
    earlier steps — e.g. compile validates what a prior write_source reported.
    """
    msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": c.id or f"call_{c.name}",
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in calls
            ],
        }
    ]
    for c in calls:
        call_id = c.id or f"call_{c.name}"
        try:
            args = json.loads(c.arguments or "{}")
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            args = {}
        try:
            tool = scenario.tool(c.name)
            result_text = tool.execute(args, history or [])
        except ScenarioError:
            result_text = f"<unknown tool: {c.name}>"
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": result_text})
    return msgs


async def _run_scenario(client: LLMClient, entry: Target, scenario: Scenario) -> ProbeResult:
    """One scripted session: a **tool loop**, not a fixed 2-turn
    script — or, for a tool-less scenario (the plain `pingpong` probe),
    a single turn.

    - With tools: turn 1 forces `required_tool`; subsequent turns keep the
      tools available but let the model choose (call again or answer).
    - While the model emits tool calls, every call — parallel or spread over
      turns — is executed by the scenario's mock and handed back (thinking
      models like qwen-3.8-27b read one file per turn).
    - The session ends on a non-empty visible `content` answer; `reasoning_
      content` alone doesn't count. `max_turns` caps the loop (else loss).

    Returns the probe dict (with `turns`); raises on transport loss of the
    forced turn 1 only — later-turn transport errors become a loss.
    """
    messages: list[dict[str, Any]] = []
    if scenario.system_prompt:
        messages.append({"role": "system", "content": scenario.system_prompt})
    messages.append({"role": "user", "content": scenario.user_prompt})
    tools = [t.schema() for t in scenario.tools]
    # None for the tool-less plain probe; __post_init__ guarantees that a
    # scenario with tools names one of them as required_tool.
    forced_tool = scenario.required_tool

    ttft: float | None = None
    e2e = 0.0
    output_tokens = 0
    chars = 0
    required_called = not forced_tool  # nothing to force for the plain probe
    used: set[str] = set()
    turns = 0

    def _result(ok: bool, error: str | None) -> ProbeResult:
        return {
            "ok": ok,
            "error": error,
            "ttft_ms": ttft,
            "e2e_ms": e2e,
            "output_tokens": output_tokens,
            "chars": chars,
            "turns": turns,
        }

    for turn in range(1, scenario.max_turns + 1):
        forced = turn == 1 and forced_tool is not None
        choice = {"type": "function", "function": {"name": forced_tool}} if forced else None
        try:
            r = await client.chat(
                entry.model,
                messages,
                stream=True,
                max_tokens=scenario.turn_max_tokens,
                tools=tools or None,
                tool_choice=choice,
                extra_body=entry.extra_body,
            )
        except RequestError as e:
            if forced:
                raise
            return _result(False, f"turn {turn} failed: {e}")
        except Exception as e:  # any transport-adjacent failure is a loss
            if forced:
                raise RequestError(str(e)) from e
            return _result(False, f"turn {turn} failed: {e}")

        turns = turn
        if ttft is None:
            ttft = r.ttft_ms
        e2e += r.e2e_ms or 0.0
        output_tokens += r.output_tokens or 0
        chars += _len_chars(r.content, r.reasoning_content)

        if r.tool_calls:
            if not required_called:
                if any(c.name == forced_tool for c in r.tool_calls):
                    required_called = True
                else:
                    names = [c.name for c in r.tool_calls]
                    return _result(
                        False,
                        f"{forced_tool} not called (got: {names or 'none'})",
                    )
            used.update(c.name for c in r.tool_calls)
            messages.extend(_tool_continuation(scenario, r.tool_calls, history=messages))
            continue

        # No tool calls → this turn should be the final answer. Reasoning
        # alone doesn't count: thinking models can finish a thinking pass and
        # stop with empty `content` (`finish_reason: "stop"`), streamed
        # reasoning included. A non-empty answer must also pass the
        # scenario's fuzzy fact checks.
        if not required_called:
            return _result(False, f"{forced_tool} not called (got: none)")
        if scenario.required_tools:
            missing = [t for t in scenario.required_tools if t not in used]
            if missing:
                return _result(False, "required tools not used: " + ", ".join(missing))
        if scenario.required_success:
            results = tool_results(messages)
            for name in scenario.required_success:
                res = results.get(name) or {}
                if not res.get("ok"):
                    detail = res.get("error") or res.get("output") or "no report"
                    return _result(False, f"{name} did not succeed: {detail}")
        if r.content.strip():
            if scenario.answer_checks:
                missing = check_answer(r.content, scenario.answer_checks)
                if missing:
                    return _result(False, "answer missing: " + ", ".join(missing))
            return _result(True, None)
        if forced_tool is None:
            # Plain probe: reasoning alone is still proof of life (chars
            # counts it); only the fully-empty reply — the thinking-starved
            # `tokens=N chars=0 ttft=None` signature — is a loss.
            if chars > 0:
                return _result(True, None)
            return _result(False, "empty response (0 chars)")
        return _result(False, f"no final answer after {forced_tool}")

    # turn cap reached without a final answer
    tail = f" after {forced_tool}" if forced_tool else ""
    return _result(False, f"no final answer{tail} (max_turns={scenario.max_turns} reached)")


async def run_ping(
    client: LLMClient,
    entries: list[Target],
    *,
    count: int | None,
    interval: float,
    on_sample: Callable[[Sample], None] | None = None,
    on_round: Callable[[int], None] | None = None,
) -> tuple[list[PingStats], bool]:
    """Run all rounds on the fixed cadence. Returns (stats, interrupted).
    count=None runs until interrupted (→ exit 130 by the caller).
    on_round is called with the round's seq before its requests fire (used
    to attribute traffic-dump records per seq).

    A Ctrl-C under `asyncio.run` (3.11+) arrives as `asyncio.CancelledError`
    on the main task — the Runner's SIGINT handler cancels it — *not* as
    `KeyboardInterrupt`, which is only re-raised if the CancelledError
    escapes. Absorbing it here keeps the partial stats so the caller can
    still print the summary (and the JSON report on stdout) before
    exiting 130."""
    results: list[PingStats] = [PingStats(model=m.model) for m in entries]
    interrupted = False
    seq = 0
    round_t0 = time.perf_counter()
    try:
        while True:
            if count is not None and seq >= count:
                break
            if on_round is not None:
                on_round(seq)
            for idx, entry in enumerate(entries):
                try:
                    # Every target runs a scenario — the plain probe is the
                    # default `pingpong` scenario, so there is one code path.
                    probe = await _run_scenario(client, entry, load_scenario(entry.scenario))
                    sample = Sample(
                        model=entry.model,
                        seq=seq,
                        ok=probe.get("ok", True),
                        ttft_ms=probe.get("ttft_ms"),
                        e2e_ms=probe.get("e2e_ms"),
                        output_tokens=probe.get("output_tokens"),
                        chars=probe.get("chars"),
                        error=probe.get("error"),
                        turns=probe.get("turns"),
                    )
                except RequestError as e:
                    sample = Sample(model=entry.model, seq=seq, ok=False, error=str(e))
                results[idx].samples.append(sample)
                if on_sample is not None:
                    on_sample(sample)
            seq += 1
            wait = round_t0 + seq * interval - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
    except (KeyboardInterrupt, asyncio.CancelledError):
        # CancelledError: SIGINT cancelled the main task (asyncio.run on
        # 3.11+); KeyboardInterrupt: delivered directly (older/lookup paths).
        # Swallowing a cancellation is deliberate — the run is over and the
        # stats collected so far must survive. Acknowledge the request so
        # asyncio does not still consider the task cancelled (the docs'
        # uncancel() contract for suppressing a cancellation).
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
        interrupted = True
    return results, interrupted
