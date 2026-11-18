# AGENTS.md

Operating notes for working on llmping (a `ping` CLI for OpenAI-compatible
LLM servers). `README.md` covers usage; this file covers how the code is
actually wired and the conventions to keep.

## Layout (the non-obvious parts)

- src-layout package (`src/llmping/`, 8 modules); CLI entrypoint
  `llmping.cli:main` (typer app) — **single command, no subcommand**
  (`invoke_without_command` callback; the app *is* the command).
- `tests/mockserver/` is the testing backbone: tests hit it as a *real* HTTP
  server (uvicorn in a thread, see `tests/conftest.py`), on an ephemeral
  port. It must stay deterministic and GPU-free. It lives in `tests/`, not
  `src/`, because the test suite is its only consumer — it is **not** shipped
  in the installed package.
- `runs/` holds `--dump-traffic` output and is gitignored.
- `rundir.py` holds run-dir creation and `TrafficDump`
  (+ `redact_headers`). No manifest/report files are written —
  the JSON report is stdout-only.

## Config — one file

Config is **one file**: `providers.yaml` (gitignored) holds named endpoints +
auth only (`providers:` list of `name` / `base_url` / `token`). There is
no `config.yaml` — what to ping is given entirely on the command line.

- A **target** = `--model MODEL` against the run's one provider, plus the
  scenario to run (`--scenario NAME`, default `pingpong`), optional
  `--extra-body JSON`, and optional `--pad N` prompt shaping.
  The CLI builds a slim `Target(model, scenario, extra_body, padding)`
  — no name, no provider field
  (`src/llmping/config.py`, plus a wrap validator so any pydantic failure
  surfaces as `ConfigError` → exit 2).
- One run = one provider + **one model**: `--provider NAME` (optional —
  resolution order `--provider` > `$LLMPING_PROVIDER` (typer `envvar`) >
  the provider named `default` in providers.yaml) picks the endpoint/auth;
  `--model MODEL` names the single model to ping (a repeat is exit 2 —
  run one llmping per model; the JSON report's `stats` block is flat, not
  keyed by model). The CLI builds **one client per run**
  (cli `_client(prov, timeout, …)`, running
  `providers.by_name(provider)`).
- Providers discovery: `--providers PATH` → `$LLMPING_PROVIDERS` →
  `./providers.yaml` (if it exists) → `~/.config/llmping/providers.yaml`
  (`$XDG_CONFIG_HOME` honoured; `discover_providers` in config.py). Load
  with `load_providers()` +
  `ProvidersConfig.by_name(provider)` (cli `_load_providers(explicit)`).
- **Connection secrets are never CLI flags** — there is no `--base-url` /
  `--token` (they would leak into shell history and `ps`). A
  `base_url` missing from YAML comes from
  `$LLMPING_BASE_URL_<NAME>` (name uppercased, non-alnum → `_`) else
  `$LLMPING_BASE_URL`; a missing `token` from `$LLMPING_TOKEN_<NAME>` else
  `$LLMPING_TOKEN` (`src/llmping/config.py`, env resolution in
  `ProviderEntry._env_fallback`). Explicit YAML always wins over env
  — including an explicit empty `token: ""` ("no auth", header omitted).
- Models are **chat-only** — there is no `type:` field.
- Unknown keys are rejected everywhere except `extra_body` (typo
  protection) → exit 2.
- **`--scenario <name>`** (optional, names the run's scenario)
  is a **registry name of a code-defined scenario**, not a file path —
  scenarios live in `llmping/scenario.py` (`Tool` + `Scenario` +
  `SCENARIOS`/`load_scenario`) because tool use needs executable mocking and
  validation that YAML can't express. Built-ins: `pingpong` (the default
  plain probe: one turn, no tools, any non-empty reply counts),
  `agentic-basic` (read mocked files, answer fuzzy facts) and `agentic`
  (read a buggy source → write the fix → compile → run; `required_tools` +
  `required_success` gates). A **bare `--scenario`** (no value, anywhere on
  the command line) lists the valid names and exits 0 — a purely local
  query (scenarios are code-defined: no providers file, no `--provider`)
  that answers like `--help` does: it wins regardless of what else is on
  the command line, and the flags it makes irrelevant (`--model`,
  `--provider`, `--count`, …) are warned about, not errors
  (`--format json` prints
  `{"scenarios": [...]}`; implemented in `_ScenarioListGroup.parse_args`
  because typer's vendored click rejects a valueless option before any
  callback runs). Unknown scenario name ⇒ exit 2 (validated up
  front in the CLI). The plain probe shares the scenario code path:
  `run_ping` always resolves `Target.scenario` (default `pingpong`) and runs
  `_run_scenario` — tool-less scenarios take the single-turn path there.
- **Never `.env`, never a token on the command line** — the token lives in
  `providers.yaml` or `LLMPING_TOKEN*` only; empty token ⇒ `Authorization`
  header omitted (Ollama/llama.cpp).

## Behavior you'd otherwise get wrong

- `--provider NAME` and `--model MODEL` — `--model` is **required** and
  names the run's **single** model (a repeated `--model` is exit 2 — one
  model per run, so the report's stats block is flat, not keyed by model);
  `--provider` is optional (env/`default` fallback above).
  Missing `--model` ⇒ usage error, exit 2. Two model-less query modes
  need no `--model`: **`llmping --list-models`** — GET
  `{base_url}/models` via `client.models()` (`_list_models` in cli.py),
  print the advertised model IDs, exit 0. No ping, no run dir, no `--model`
  (combining `--list-models` with `--model` → exit 2); run-only flags
  (`--scenario`, `--extra-body`, `--count`, `--interval`, `--runs-dir`,
  `--dump-traffic`) are ignored with a stderr warning. A failed
  request (timeout/connect/HTTP) prints the error and exits 1, like a lost ping.
  The mock server advertises `MockOpenAIServer.DEFAULT_MODELS`
  (`gpt-oss-120b`, `mistral-small-4-119b`) on `GET /v1/models` — and a
  **bare `--scenario`** lists scenario names (see the scenario bullet
  above for its contract).
- Client never retries (httpx never retries by default, explicit in code
  because a retry would corrupt latency stats); errors surface as
  `RequestError` with a `status`. Timeouts: default per-request 30 s,
  connect 10 s.
- Streaming parses SSE; `usage_source` is `"server"` vs `"none"` — a cheap
  way to tell real servers (which honour `include_usage`) from wrappers.
  Ping's `tokens=N` comes from the server usage block; a server that sends
  no usage block reports no token count (the live line omits `tokens=` — a
  chunk count is not a token count).
- Mock server timing is ground truth: `LLMPING_MOCK_FIRST_CHUNK_MS` /
  `LLMPING_MOCK_CHUNK_MS`. Model-behaviour presets (comma-separated model
  names via `LLMPING_MOCK_*`): `FAIL` → 503s, `THINK_ONLY` → reasoning with
  no visible answer, `SERIAL` → one file read per tool turn (qwen),
  `EMPTY` → HTTP 200 with 0 content AND 0 reasoning (the empty-response
  loss), `WRONG_ANSWER` → plausible text that fails the scenario's fact checks.
- Conftest exposes one fixture per preset (`mock_server`, `mock_server_failing`,
  `mock_server_think_only`, `mock_server_serial_reader`, `mock_server_empty`,
  `mock_server_wrong_answer`) — reuse them; don't hand-roll a server in a test.
  For a non-standard preset use `mock_server_factory(**preset_kwargs)`; every
  server it starts is stopped at teardown. The agentic-basic mock
  replays the *unread* mock files (`MOCK_FILES`: `src/fib.py` + `README.md`)
  from prior `tool` results, so a tool loop terminates once both are read.
  The mock also plays the whole `agentic` pipeline automatically when the
  toolbox contains `write_source` (one ideal one-step-per-turn tool call:
  read → write FIXED → compile → run → answer).
- `--format json` prints only the report JSON on stdout; status/warnings go
  to stderr. The report carries `provider`, `model`, `scenario`, `stats`
  (flat, not keyed by model — one model per run),
  `duration_s`, `timestamp` (run start, ISO-8601 UTC) and `base_url` —
  there is **no `command` field**. **Nothing is written to disk**: the
  report is stdout-only under `--format json`. The **table** summary is a labeled
  block — `--- ping statistics ---` header, then `provider:`, `model:`,
  `scenario:`, `target: <base_url>` lines, then the sent/received/loss
  line and the e2e/ttft min/avg/max/mdev lines.
- **Ctrl-C on `--count inf`** arrives as `asyncio.CancelledError` on the
  main task under `asyncio.run` (its SIGINT handler cancels it) — *not* as
  `KeyboardInterrupt`, which is only re-raised if the CancelledError escapes.
  `run_ping` catches both, swallows the cancel deliberately, and returns
  `(partial_stats, interrupted=True)` so the CLI still prints the summary
  (and the JSON report on stdout) and exits 130.
- **No subcommand** — the CLI is one command (typer callback with
  `invoke_without_command`): `llmping --providers X --dump-traffic --provider
  p --model m` — every flag is an option on that one command; there is no
  `ping --providers X` form. Short forms: `-p` provider, `-m` model,
  `-s` scenario (bare `-s`, like bare `--scenario`, lists the names),
  `-c` count, `-i` interval, `-t` timeout, `-d` dump-traffic, `-f` format,
  `-l` list-models, `-V` version, `-h` help
  (`context_settings={"help_option_names": ["-h", "--help"]}`).
- Run dirs exist **only for `--dump-traffic`**: without it a run writes
  nothing to disk.
  `runs/<provider>_<model>_<scenario>_<ts>/`, created only when the dump
  is on, holding the exchange log split **per round/seq**:
  `traffic-<seq>.jsonl` (one timestamped, kind-tagged JSON line per
  exchange; bodies raw AND decoded, auth headers masked) — handy for
  inspecting actual SSE from proxies. `/` (and `\`, NUL) in
  provider/model/scenario names is escaped to `_` (`RunDir._escape_name`)
  so a name can't break out of the runs dir; a same-second collision
  appends `-1`, `-2`, … (atomic `mkdir`, so concurrent runs never share or
  overwrite a dir). `begin()` closes
  the previous seq's handle, so at most one dump file is open at a time (a
  `--count inf` run must not leak one fd per round).
- `chars=` counts content + reasoning content, summed across all turns for
  scenarios. Thinking models (e.g. the qwen model) emit **only**
  `reasoning_content` deltas, so `content` is empty and content-only lengths
  would read 0. TTFT is taken from the first content *or* reasoning delta.
- An **empty plain reply** — HTTP 200 with zero `content` AND zero
  `reasoning_content` (the thinking-starved signature: `tokens=N chars=0`,
  ttft missing) — is a **loss** (`empty response (0 chars)`), not a reply;
  reasoning-only streams still count. The probe's `max_tokens` is a **cap,
  not a spend**: `PROBE_MAX_TOKENS = 4096` — classic models still stop right
  after "pong" so probes stay cheap, and thinking models get headroom to
  answer (a tiny budget starves them into the empty reply; the scenario
  `turn_max_tokens` is separate, 256). `--extra-body '{"max_tokens": N}'`
  can still override it — the client sets `max_tokens` first, then
  `payload.update(extra_body)`.
- Scenario sessions (`--scenario NAME`; every run executes one, default
  `pingpong`) — each round is a
  **tool loop, not a fixed 2-turn script** (the tool-less `pingpong` probe
  is the single-turn special case of the same loop):
  - Turn 1 forces the required tool (`tool_choice`); later turns keep the
    tools available with auto `tool_choice`, so the model either calls a
    tool again or answers (`max_turns` default 6 caps it, 8 for
    `agentic`). A tool-less scenario forces nothing and needs no tools.
  - **Parallel and multi-turn tool calls are both handled**: every call the
    model emits — batched (gemma-4-31b read `src/fib.py` + `README.md` in
    one turn) or spread over turns (qwen-3.8-27b reads one file per turn) —
    gets its own mocked result handed back.
  - Metrics: `ttft` = turn-1 first token; `e2e` spans the whole session;
    `tokens`/`chars` sum all turns; live line + report carry `turns=`;
    `input_tokens` sums the server-counted prompt size of every turn
    (reported as `in=` on the live line and `input_tokens` in the stats).
  - **`--pad N`** appends N inert characters to the first user message of
    any scenario (growing every later turn with it) — used to exercise
    size-dependent behavior/token accounting; `Target.padding` (≥ 0,
    invalid → exit 2), applied in `_run_scenario` via `_pad`.
  - Round ok requires, in order: required tool called (tool-less scenarios
    skip this); **non-empty visible `content`** final answer
    (`reasoning_content` alone doesn't count — thinking models stop after a
    thinking pass with empty `content`; the tool-less `pingpong` probe is the
    one exception: reasoning alone still counts as a reply, only the
    fully-empty stream is the `empty response (0 chars)` loss); the
    answer passes the scenario's **fuzzy fact checks**
    (`Scenario.answer_checks` via `check_answer` — agentic-basic needs
    `fib`/`fibonacci` and `42`/`forty-two`, case-insensitive substring),
    else `answer missing: …`; `Scenario.required_tools` — all pipeline
    tools called (agentic: `read_source`→`write_source`→`compile`→`run`;
    reading alone fails → `required tools not used: …`);
    `Scenario.required_success` — the **latest** mocked result of
    compile/run must report `ok: true` (writing the buggy source back
    fails the round even if the answer names the output).
  - Executors take `(args, history)` and stay stateless: they read earlier
    tool results via `tool_results(messages)` (correlates assistant
    `tool_calls` ids with `tool` messages) — `compile` validates what the
    prior `write_source` reported written; `run` only succeeds after
    `compile` ok. A server that ignores forced `tool_choice` loses every
    round (exit 1). `--dump-traffic` writes one exchange per turn, so a
    `traffic-<seq>` file holds one line per turn.

## Commands on this box

System pip is PEP 668-blocked and `uv` is not installed — use the checkout
venv (`.venv`, already set up). Prefer the `.venv/bin/` forms so commands
don't depend on activation state:

```bash
.venv/bin/pip install -e ".[dev]"     # deps (once)
.venv/bin/python -m pytest -q           # whole suite (~40s, no GPU)
.venv/bin/python -m pytest tests/test_client.py::test_name -q   # one test; add -k to filter
```

Gate — all four, verified in order, before committing:

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src                    # [tool.mypy] files = ["src"]
.venv/bin/python -m pytest
```

Coverage is informational, not a gate:
`.venv/bin/python -m pytest --cov=llmping --cov-report=term-missing`.

`tests/test_cli.py` has an autouse fixture that chdirs into `tmp_path`, so
the CLI's default `--runs-dir ./runs` lands there and a suite run leaves
no run dirs in the repo working tree.

Standalone loopback target for manual CLI runs:

```bash
.venv/bin/python -m tests.mockserver.serve --port 8321
# providers.yaml base_url needs the /v1 suffix: http://127.0.0.1:8321/v1
```

## Commits

Add a git pseudo-header when a commit is agent-authored:
`AI-assisted: yes, <agent> + <model>`.
