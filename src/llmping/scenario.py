"""Scenarios for `ping`: scripted mini sessions with mocked tools.

Scenarios are defined in **code**, not YAML — file reading here is a mocked
tool executor with a tiny virtual filesystem (no real I/O), and the pattern
extends to arbitrarily complex tool use later. A run picks one by name
(`--scenario`, a registry name, not a file path; the CLI stores it on the
target as `Target.scenario`); `ping` runs that scripted session. The
registry also holds `pingpong`, the default plain probe (one turn, no
tools) — the probe is "just" the tool-less scenario.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ScenarioError(ValueError):
    """Unknown/misdeclared scenario → CLI exit 2."""


# Conversation messages as handed to the model (system/user/assistant/tool dicts).
History = list[dict[str, Any]]


# The plain probe is the default scenario (`pingpong`): demand the
# single word "pong" and take any non-empty reply as proof of life.
CHAT_PROMPT = "Reply with the single word: pong."

# max_tokens is a CAP, not a spend target: classic models still stop right
# after "pong" (~2 tokens) so probes stay cheap, but a tiny budget starves
# *thinking* models (qwen-3.8-27b spends everything on reasoning and then
# emits an empty reply → the empty-response loss). Headroom fixes that.
PROBE_MAX_TOKENS = 4096

# The scenario a target runs when `--scenario` isn't given.
DEFAULT_SCENARIO = "pingpong"


@dataclass(frozen=True)
class Tool:
    """A tool the model may call, with a mocked executor (no real I/O).

    `execute(args, history)` runs the mock and returns the result text fed
    back to the model. `history` is the conversation so far (prior assistant
    `tool_calls` + `tool` results), which lets executors reason about earlier
    steps — e.g. a mocked compiler validates the source handed back by a
    prior `write_source` call, with no real I/O and no shared state.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any], History | None], str]  # (args, history) → result text

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# (fact label, accepted substrings — any one matching the answer counts).
# Fuzzy on purpose: non-deterministic models phrase the same fact differently,
# so we match loosely on the actual values, not on exact sentences.
AnswerCheck = tuple[str, tuple[str, ...]]


def check_answer(text: str, checks: tuple[AnswerCheck, ...]) -> list[str]:
    """Return the labels of facts NOT found in `text` (case-insensitive
    substring). Empty list = the answer covers every required fact."""
    lower = text.lower()
    missing = []
    for label, tokens in checks:
        if not any(tok in lower for tok in tokens):
            missing.append(label)
    return missing


def tool_results(messages: History) -> dict[str, dict[str, Any]]:
    """Map tool name → parsed result of its *latest* call, by correlating
    assistant `tool_calls` (id → name) with `tool` messages (tool_call_id).
    Lets a mocked executor (e.g. compile) see what an earlier step produced."""
    ids: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            ids[tc.get("id") or ""] = (tc.get("function") or {}).get("name") or ""
        if m.get("role") == "tool":
            name = ids.get(m.get("tool_call_id") or "")
            if name:
                try:
                    results[name] = json.loads(m.get("content") or "{}")
                except (ValueError, TypeError):
                    results[name] = {}
    return results


@dataclass(frozen=True)
class Scenario:
    """A scripted mini session: the model must call `required_tool`,
    the mocked executor returns a result, and the model produces a final
    answer. This is a **loop**, not a fixed 2-turn script: while the model
    keeps emitting tool calls they are executed and handed back (a thinking
    model may read one file per turn, e.g. qwen-3.8-27b), up to `max_turns`.
    `ttft` is the turn-1 first token (the tool call); `e2e` spans the whole
    session (all turns + mocked exec); `turns` counts model turns used.
    A **tool-less** scenario (`required_tool=None`, no `tools`) is the plain
    probe: a single turn where any non-empty reply counts.
    `answer_checks` (optional) fuzzy-validates the final answer: every fact
    must be loosely present, else the round is a loss. `required_tools`
    (optional, independent of the turn-1 gate) is the full set of tools
    that must ALL have been called by
    session end; a missing step is a loss
    — for a pipeline like read → write → compile → run, this enforces the
    whole sequence, not just the first call. `required_success` (optional)
    lists tools whose **latest** mocked result must report `ok: true` — so a
    round only counts if e.g. compile AND run actually succeeded, not merely
    that the tools were invoked."""

    name: str
    description: str
    system_prompt: str
    user_prompt: str
    tools: tuple[Tool, ...]
    required_tool: str | None = None
    turn_max_tokens: int = 256
    max_turns: int = 6
    required_tools: tuple[str, ...] = ()
    required_success: tuple[str, ...] = ()
    answer_checks: tuple[AnswerCheck, ...] = ()

    def __post_init__(self) -> None:
        # The engine forces `required_tool` on turn 1, so a scenario with
        # tools must name one; a scenario without tools is the plain probe.
        if self.tools and not self.required_tool:
            raise ScenarioError(f"scenario {self.name!r}: has tools but no required_tool")
        if self.required_tool and self.required_tool not in {t.name for t in self.tools}:
            raise ScenarioError(
                f"scenario {self.name!r}: required_tool {self.required_tool!r} "
                "is not one of its tools"
            )

    def tool(self, name: str) -> Tool:
        for t in self.tools:
            if t.name == name:
                return t
        raise ScenarioError(f"scenario {self.name!r} has no tool {name!r}")


# --------------------------------------------------------------------------- built-in scenarios


def _read_file_mock(args: dict[str, Any], history: History | None = None) -> str:
    """Mocked file read — a tiny virtual filesystem. Explicitly no real I/O."""
    path = str(args.get("path") or "src/fib.py")
    files = {
        "src/fib.py": "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
        "config.yaml": "host: localhost\nport: 8321\n",
        "README.md": "# demo\n\nThe magic constant is 42.\n",
    }
    return json.dumps({"path": path, "content": files.get(path, f"<mocked file: {path}>")})


READ_FILE_TOOL = Tool(
    name="read_file",
    description=(
        "Read the contents of a file in the repository. Use it to satisfy file-based questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, e.g. src/fib.py",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_read_file_mock,
)


# --- agentic: read a buggy source, write the fix, compile it, run it
# The mocked toolchain is *stateful via history* (no real I/O, no shared
# state): compile validates the source a prior write_source reported written,
# and run only succeeds once compile reported ok — so a round genuinely needs
# the read → write → compile → run pipeline to have worked.

BUGGY_HELLO = 'def greet():\n    pritn("Hello, world!")\n\ngreet()\n'
FIXED_HELLO = 'def greet():\n    print("Hello, world!")\n\ngreet()\n'

HELLO_PATH = "src/hello.py"


def _read_source_mock(args: dict[str, Any], history: History | None = None) -> str:
    return json.dumps({"path": str(args.get("path") or HELLO_PATH), "content": BUGGY_HELLO})


def _write_source_mock(args: dict[str, Any], history: History | None = None) -> str:
    path = str(args.get("path") or HELLO_PATH)
    content = str(args.get("content") or "")
    # echo the content so a later compile can validate what was written
    return json.dumps({"path": path, "written": True, "bytes": len(content), "content": content})


def _compile_mock(args: dict[str, Any], history: History | None) -> str:
    written = (tool_results(history or []).get("write_source") or {}).get("content") or ""
    fixed = "pritn" not in written and "print(" in written
    if fixed:
        return json.dumps({"ok": True, "detail": "compiled cleanly"})
    return json.dumps({"ok": False, "error": "NameError: name 'pritn' is not defined"})


def _run_mock(args: dict[str, Any], history: History | None) -> str:
    compiled = (tool_results(history or []).get("compile") or {}).get("ok")
    if compiled:
        return json.dumps({"ok": True, "output": "Hello, world!"})
    return json.dumps({"ok": False, "error": "program did not compile"})


READ_SOURCE_TOOL = Tool(
    name="read_source",
    description="Read the contents of a source file in the repository.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file, e.g. src/hello.py"}
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_read_source_mock,
)

WRITE_SOURCE_TOOL = Tool(
    name="write_source",
    description="Write the corrected contents of a source file to the repository.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to write"},
            "content": {"type": "string", "description": "The full corrected file contents"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    execute=_write_source_mock,
)

COMPILE_TOOL = Tool(
    name="compile",
    description="Compile (syntax-check) a file in the repository and report whether it compiles.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the file to compile"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_compile_mock,
)

RUN_TOOL = Tool(
    name="run",
    description="Run a compiled program in the repository and report its output.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path of the program to run"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_run_mock,
)

SCENARIOS: dict[str, Scenario] = {
    "pingpong": Scenario(
        name="pingpong",
        description=(
            "The default plain probe: one turn, no tools — the prompt demands "
            "the single word 'pong', and any non-empty reply (visible content "
            "or reasoning) counts."
        ),
        system_prompt="",
        user_prompt=CHAT_PROMPT,
        tools=(),
        required_tool=None,
        turn_max_tokens=PROBE_MAX_TOKENS,
    ),
    "agentic-basic": Scenario(
        name="agentic-basic",
        description=(
            "The model must call read_file to fetch a file, then answer a question about it."
        ),
        system_prompt=(
            "You are a code reviewer. When asked about a file, you MUST call the "
            "read_file tool first and base your answer on its contents."
        ),
        user_prompt=(
            "What function does src/fib.py define, and what does the constant in "
            "README.md tell you?"
        ),
        tools=(READ_FILE_TOOL,),
        required_tool="read_file",
        answer_checks=(
            ("a mention of the fib function", ("fib", "fibonacci")),
            ("the magic constant 42", ("42", "forty-two", "forty two")),
        ),
    ),
    "agentic": Scenario(
        name="agentic",
        description=(
            "The model must read a buggy hello-world source, write the fixed version, "
            "compile it, and run it."
        ),
        system_prompt=(
            "You are a software engineer fixing a buggy Python program. Work through it "
            "with the tools in this order: read_source to see the bug, write_source to "
            "write the corrected file, compile to verify it, run to execute it. Base your "
            "final answer on the tool results."
        ),
        user_prompt=(
            "src/hello.py is buggy and does not run. Read it, write the corrected version, "
            "compile it, then run it. Report what the fixed program prints."
        ),
        tools=(READ_SOURCE_TOOL, WRITE_SOURCE_TOOL, COMPILE_TOOL, RUN_TOOL),
        required_tool="read_source",
        required_tools=("read_source", "write_source", "compile", "run"),
        required_success=("compile", "run"),
        max_turns=8,
        answer_checks=(
            ("the program output", ("hello, world!",)),
            ("evidence of the fix", ("print", "pritn", "fixed", "corrected", "typo")),
        ),
    ),
}


def load_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise ScenarioError(f"unknown scenario {name!r} — known: {known}") from None
