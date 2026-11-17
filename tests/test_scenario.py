"""Scenario registry + mocked tool tests."""

from __future__ import annotations

import json

import pytest

from llmping.scenario import (
    BUGGY_HELLO,
    COMPILE_TOOL,
    FIXED_HELLO,
    HELLO_PATH,
    READ_FILE_TOOL,
    READ_SOURCE_TOOL,
    RUN_TOOL,
    WRITE_SOURCE_TOOL,
    ScenarioError,
    check_answer,
    load_scenario,
)


def test_load_scenario_agentic_basic():
    sc = load_scenario("agentic-basic")
    assert sc.name == "agentic-basic"
    assert sc.required_tool == "read_file"
    assert [t.name for t in sc.tools] == ["read_file"]
    assert sc.tool("read_file").name == "read_file"
    # the scenario fuzzy-validates the final answer
    assert len(sc.answer_checks) == 2


def test_check_answer_fuzzy_facts():
    checks = load_scenario("agentic-basic").answer_checks
    # exact-ish answer passes every fact
    ok = "src/fib.py defines the fib(n) Fibonacci function; README says 42."
    assert check_answer(ok, checks) == []
    # alternate phrasings + case-insensitivity still pass (fuzzy by design)
    assert check_answer("FIBONACCI ... FORTY-TWO", checks) == []
    assert check_answer("Fibonacci ... forty two", checks) == []
    # generic "I read the files" replies miss both facts
    missing = check_answer("I read the files and everything looks fine.", checks)
    assert missing == ["a mention of the fib function", "the magic constant 42"]
    # half an answer still fails that half
    assert "a mention of the fib function" in check_answer("42 is the magic constant.", checks)
    assert "the magic constant 42" in check_answer("it defines fib(n).", checks)
    # no checks → nothing to miss
    assert check_answer("anything", ()) == []


def test_load_unknown_scenario():
    with pytest.raises(ScenarioError, match="unknown scenario"):
        load_scenario("nope")


def test_read_file_mock_returns_known_file():
    out = json.loads(READ_FILE_TOOL.execute({"path": "src/fib.py"}))
    assert out["path"] == "src/fib.py"
    assert "def fib" in out["content"]


def test_read_file_mock_defaults_and_fallback():
    # args without a path fall back to a default file (mocked server shapes
    # don't always pass a real path); unknown paths get a canned placeholder.
    out = json.loads(READ_FILE_TOOL.execute({}))
    assert "fib" in out["path"]
    out2 = json.loads(READ_FILE_TOOL.execute({"path": "missing.txt"}))
    assert "missing.txt" in out2["content"]
    assert "mocked file" in out2["content"]


def test_tool_schema():
    s = READ_FILE_TOOL.schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "read_file"
    assert "path" in s["function"]["parameters"]["required"]


# --- agentic: read buggy source → write fix → compile → run ---------------


def _tool_history(*steps: tuple[str, str, str]) -> list[dict]:
    """Conversation shaped like ping._tool_continuation's output: an assistant
    tool_calls message + its `tool` result, per step (name, args-json, result)."""
    msgs: list[dict] = []
    for i, (name, args_json, result) in enumerate(steps):
        cid = f"call_{name}_{i}"
        tool_call = {
            "id": cid,
            "type": "function",
            "function": {"name": name, "arguments": args_json},
        }
        msgs.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
    return msgs


def _write_step(content: str) -> list[dict]:
    args = json.dumps({"path": HELLO_PATH, "content": content})
    result = json.dumps(
        {"path": HELLO_PATH, "written": True, "bytes": len(content), "content": content}
    )
    return _tool_history(("write_source", args, result))


def test_load_scenario_agentic():
    sc = load_scenario("agentic")
    assert sc.name == "agentic"
    assert sc.required_tool == "read_source"
    assert [t.name for t in sc.tools] == ["read_source", "write_source", "compile", "run"]
    # the full pipeline must be called, and compile+run must actually succeed
    assert sc.required_tools == ("read_source", "write_source", "compile", "run")
    assert sc.required_success == ("compile", "run")
    assert sc.max_turns == 8
    assert len(sc.answer_checks) == 2


def test_agentic_read_source_returns_buggy_file():
    out = json.loads(READ_SOURCE_TOOL.execute({"path": HELLO_PATH}))
    assert out["path"] == HELLO_PATH
    assert "pritn" in out["content"]  # the bug (would NameError at runtime)
    assert "print(" not in out["content"]


def test_agentic_write_source_echoes_content():
    out = json.loads(WRITE_SOURCE_TOOL.execute({"path": HELLO_PATH, "content": FIXED_HELLO}))
    assert out["written"] is True
    assert out["bytes"] == len(FIXED_HELLO)
    assert out["content"] == FIXED_HELLO  # echoed so compile can validate it


def test_agentic_compile_accepts_fixed_write():
    history = _write_step(FIXED_HELLO)
    out = json.loads(COMPILE_TOOL.execute({"path": HELLO_PATH}, history))
    assert out["ok"] is True
    assert "detail" in out


def test_agentic_compile_rejects_buggy_write():
    # The model "fixed" nothing — wrote the typo back. Compile must fail on it.
    history = _write_step(BUGGY_HELLO)
    out = json.loads(COMPILE_TOOL.execute({"path": HELLO_PATH}, history))
    assert out["ok"] is False
    assert "pritn" in out["error"]


def test_agentic_compile_without_any_write_fails():
    out = json.loads(COMPILE_TOOL.execute({"path": HELLO_PATH}, []))
    assert out["ok"] is False


def test_agentic_run_succeeds_after_clean_compile():
    history = _write_step(FIXED_HELLO)
    compiled = json.loads(COMPILE_TOOL.execute({"path": HELLO_PATH}, history))
    assert compiled["ok"] is True
    history += _tool_history(("compile", json.dumps({"path": HELLO_PATH}), json.dumps(compiled)))
    out = json.loads(RUN_TOOL.execute({"path": HELLO_PATH}, history))
    assert out["ok"] is True
    assert out["output"] == "Hello, world!"


def test_agentic_run_fails_without_clean_compile():
    # discipline: run must not report output if compile never succeeded.
    history = _write_step(BUGGY_HELLO)
    out = json.loads(RUN_TOOL.execute({"path": HELLO_PATH}, history))
    assert out["ok"] is False
