# llmping

`ping` for OpenAI-compatible LLM servers (vLLM, SGLang, TGI, llama.cpp,
Ollama, gateways).

- **`ping`** — repeated minimal probes, ping-style live output, loss %,
  `--count` / `--interval`, exit codes for monitoring.
- **Scenarios** — code-defined mini sessions run by `llmping`
  where the model must call tools (e.g. `read_file`, or the `agentic`
  read→write→compile→run pipeline) whose execution is
  mocked (no real I/O); pick one with `--scenario <name>`. The default
  `pingpong` scenario is the plain probe (one turn, no tools).
- Run dirs exist **only for `--dump-traffic`**: `runs/<provider>_<model>_
  <scenario>_<ts>/` holds the raw wire traffic (one timestamped,
  kind-tagged JSON line per HTTP exchange, bodies both raw and decoded,
  auth redacted; split per round/seq as `traffic-<seq>.jsonl`; `/` in
  names is escaped to `_`). Without `--dump-traffic` a run writes
  **nothing** to disk — the JSON report goes to stdout (`--format json`).

## Install

With [pipx](https://pipx.pypa.io) (recommended — isolated, on PATH):

```bash
pipx install git+https://github.com/lnussbaum/llmping
```

From a checkout (development):

```bash
git clone https://github.com/lnussbaum/llmping
cd llmping
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.13. System `pip` is blocked on PEP 668-managed boxes —
use pipx or a venv (or `uv` if you have it).

## Configure

One file: **`providers.yaml`** — named endpoints + auth (gitignored; tokens
live here, never `.env`). There is **no `config.yaml`**: what to ping is
given on the command line (`--provider NAME --model MODEL`). Copy
`providers.example.yaml` and edit:

```yaml
# providers.yaml
providers:
  - name: local-vllm
    base_url: http://localhost:8000/v1
    token: ""                 # empty ⇒ Authorization header omitted
  - name: prod
    base_url: https://api.example.fr/v1
    token: s3cr3t
```

The `--model MODEL` names the run's single model on that provider; every
run executes a **scenario** named in code (`src/llmping/scenario.py`),
chosen with `--scenario NAME` — the default `pingpong` is the plain probe
(one turn, no tools). Scenarios are defined in code, not YAML, and their
tool execution is mocked. There is no `models:` registry and no `targets:`
list — a target is entirely a command line.

Providers discovery: `--providers PATH` → `$LLMPING_PROVIDERS` →
`./providers.yaml` → `~/.config/llmping/providers.yaml`
(`$XDG_CONFIG_HOME` honoured).

Connection details can also come from the environment when `providers.yaml`
leaves them out: per-provider `LLMPING_BASE_URL_<NAME>` /
`LLMPING_TOKEN_<NAME>` (name uppercased, non-alphanumeric → `_`) override the
global `LLMPING_BASE_URL` / `LLMPING_TOKEN`; an explicit YAML value always
wins. `$LLMPING_PROVIDER` is the default `--provider`. There are
**no `--base-url` / `--token` flags** — secrets never appear
on a command line (they would leak into shell history and `ps`).

## Usage

```bash
llmping --provider local-vllm --model gpt-oss-120b        # 1 round of the default pingpong scenario; exit 1 on any loss
llmping --provider local-vllm --model gpt-oss-120b --count 10 --interval 1.0
llmping --provider prod --model m1 --count inf # until Ctrl-C (exit 130); one llmping per model
llmping --scenario                                     # list the valid scenario names, then exit
llmping --provider prod --model gpt-oss-120b --scenario agentic-basic   # agentic: forced read_file (mocked) → answer
llmping --provider prod --model deepseek-v4-flash-0731 --scenario agentic  # read→write→compile→run pipeline (mocked)
llmping --provider prod --model gpt-oss-120b --extra-body '{"reasoning_effort": "high"}'  # extra request body
llmping --provider prod --list-models               # query mode: list the provider's advertised models, then exit
```

`--provider NAME` picks the provider; it's optional — without it the run
uses `$LLMPING_PROVIDER`, else the provider named `default` in
providers.yaml. One run pings **exactly one model** (`--model MODEL`);
repeating `--model` is a usage error — run one llmping per model.

Run flags: `--providers`, `--runs-dir`, `--dump-traffic` (log
every HTTP exchange — raw + decoded — to
`<runs-dir>/<provider>_<model>_<scenario>_<ts>/traffic-<seq>.jsonl`),
`--pad N` (N chars of inert padding appended to the first user message —
grows the prompt to exercise token accounting; input tokens are reported),
`--nonce` (prepends a random per-round nonce to the first message — busts
the server's KV/prefix cache so every round pays a true cold prefill; the
server-reported cached token count shows up as `cached=`),
`--format table|json` (JSON prints the report on stdout, status on stderr).
The JSON report carries `provider`, `model`, `scenario`, `stats`
(sent/replies/loss_pct/e2e/ttft/tokens/input_tokens/chars/turns — flat,
not keyed by model), `duration_s`,
`timestamp` (run start, ISO-8601 UTC), `base_url` and `providers_path`.
The CLI is one command — no subcommand.
Short options: `-p` provider, `-m` model, `-s` scenario, `-c` count,
`-i` interval, `-t` timeout, `-d` dump-traffic, `-f` format, `-l`
list-models, `-V` version, `-h` help
(values attach too: `-mmistral`, `-fjson`).

Exit codes: `0` ok · `1` degraded (ping loss, or a failed `--list-models` request) ·
`2` config/usage error · `130` interrupted `--count inf`.

## Development

```bash
pip install -e ".[dev]"     # in the venv from Install
pytest                      # tests run against the repo's mock server (tests/mockserver)
```

Quality gate — all four must pass before committing:
`ruff check`, `ruff format --check`, `mypy src`, `pytest`.
Coverage is informational: `pytest --cov=llmping --cov-report=term-missing`.

Tests need no GPU: they run against the deterministic mock server
(`tests/mockserver`). Run it standalone as a loopback target:

```bash
python -m tests.mockserver.serve --port 8321
```

Tune it with `LLMPING_MOCK_FIRST_CHUNK_MS`, `LLMPING_MOCK_CHUNK_MS`, and the
comma-separated model-name presets `LLMPING_MOCK_FAIL` (503),
`LLMPING_MOCK_THINK_ONLY` (reasoning, no visible answer),
`LLMPING_MOCK_SERIAL` (one file read per tool turn),
`LLMPING_MOCK_EMPTY` (HTTP 200 with no content and no reasoning) and
`LLMPING_MOCK_WRONG_ANSWER` (plausible text that fails the scenario's checks).

## License

MIT — see [LICENSE](LICENSE).
