# scripts/ — recurring probes + static HTML reports

`llmping-run.py` is a cron-friendly addon that runs llmping against a set of
targets on a schedule and turns the collected history into a static HTML
site. It lives in `scripts/` (not the installed package): llmping stays a
one-shot probe, and this wrapper only *consumes* it —
it spawns `llmping --format json` per target and never imports llmping
internals.

## Setup

Create `probes.yaml` next to the repo root (gitignored — copy the shape):

```yaml
targets:
  - name: albert-deepseek          # id → URL slug + chart label (unique)
    provider: albert              # a providers.yaml entry
    model: deepseek-v4-flash-0731
    scenario: pingpong            # optional (llmping's default)
    count: 5                      # optional per-target overrides
    interval: 1.0
    timeout: 30
    group: deepseek-providers     # optional — comparison page across targets
  - name: scaleway-deepseek
    provider: scaleway
    model: deepseek-v4-flash-0731
    group: deepseek-providers     # same group ⇒ compared on one chart
  - name: qlf-qwen-agentic
    provider: qlf-proxy-ia
    model: qwen-3.8-27b
    scenario: agentic-basic
    count: 3
```

Endpoints/auth stay in llmping's own `providers.yaml` (discovered by the
llmping subprocess: `--providers` → `$LLMPING_PROVIDERS` → `./providers.yaml`
→ `~/.config/llmping/providers.yaml`) — the wrapper never reads secrets; it
only forwards a `--providers` path when you pass one.

## Usage

```bash
python scripts/llmping-run.py probe    # run every target once, append history/
python scripts/llmping-run.py report    # scan history/, write site/
# both in one go (shared --history/--site go before the subcommand):
python scripts/llmping-run.py --history history --site site probe --report
```

Options that matter: `--providers PATH` (forwarded to llmping), `--only NAME`
(repeatable — probe one target), `--jobs N` (parallel llmping processes,
default 6), `--history DIR` / `--site DIR` (defaults: `history/`, `site/`
beside the repo root), `--days N` (report window; default: forever).

The llmping binary is found via `$LLMPING_BIN`, then the venv the script
runs from, then `PATH` — plain `python scripts/llmping-run.py` works from
cron with a bare PATH as long as the same venv has llmping installed.

## Data layout (append-only)

Each probe appends one JSONL line to `history/YYYY-MM-DD.jsonl`:

```json
{"ts": "...", "target": "albert-deepseek", "exit": 0,
 "report": { ...the full llmping --format json report... },
 "stderr": "warnings, or the error when no report was produced"}
```

Retention is forever: nothing is pruned or rewritten, and the report step
rescans everything (a torn line never breaks it). A run that produced no
report at all (crash, bad flag, probe-level timeout) is recorded with
`"report": null` — the site shows it as a **down** dot.

## Site

`site/index.html` — a card per target: status dot (last run: green = 0%
loss, amber = partial loss, red = 100%/no report), last loss, latest e2e/ttft
averages, an e2e sparkline (inline SVG, no JS). `site/<target>.html` —
loss/e2e/ttft charts over the whole window, the ten latest runs in a table,
and the last stderr when there was one. The per-target charts are rendered
client-side by [uPlot](https://github.com/leeoniya/uPlot) loaded from a CDN
(jsdelivr, pinned to 1.6.32): real time axis, grid, hover cursor — the data
is embedded in the page as JSON, so the pages are static but need internet
access when opened. The index (and everything else) works fully offline.

Every chart opens on the last 24 hours with `24h · 7d · 30d · all` chips
above it — the full history stays embedded, so switching windows is instant
(drag-zoom and double-click-reset work as always).

Targets sharing a `group:` in probes.yaml get a comparison page
(`site/group-<name>.html`): loss/e2e/ttft on shared wall-clock axes, one
line per member target (e.g. the same model across several providers),
linked from the index and from each member page's roster. Groups render
only when at least two members have data; `report` learns them from the
same probes.yaml (`--probes`, default `probes.yaml`). Group series are
**bucketed to the minute** (floor): same-round probes land in the same
bucket (within-bucket runs average), and a member that skipped a bucket is
bridged with a straight segment (uPlot `spanGaps`) — a missed round doesn't
chop the line apart.

## Cron

```
*/10 * * * *  cd /path/to/llmping && .venv/bin/python scripts/llmping-run.py probe --report --providers /path/to/providers.yaml
```

`probe` exits 1 when a target produced **no report** (losses are data, not
errors) — cron can mail on that. Every envelope records a stderr tail
(shown on the per-target page). Note that cron jitter and box load pollute
latency numbers; treat e2e/ttft as trends, not lab measurements.
