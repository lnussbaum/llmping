#!/usr/bin/env python3
"""llmping-run — cron-friendly wrapper that runs llmping against a set of
targets on a schedule and turns the collected history into a static HTML site.

    python scripts/llmping-run.py probe            # run all targets, append history
    python scripts/llmping-run.py report           # regenerate site/ from history/
    python scripts/llmping-run.py probe --report    # both

What to probe lives in probes.yaml (see scripts/README.md):

    targets:
      - name: albert-deepseek        # id → URL slug + chart label
        provider: albert            # a providers.yaml entry
        model: deepseek-v4-flash-0731
        scenario: pingpong          # optional (llmping default)
        count: 5                    # optional per-target overrides
        interval: 1.0
        timeout: 30

llmping's own contract is unchanged: this script is a *consumer* that spawns
`llmping -f json` per target (never imports llmping internals) and appends
one JSONL line per run to history/YYYY-MM-DD.jsonl:

    {"ts": ..., "target": ..., "exit": ..., "report": {...}, "stderr": ...}

History is append-only — nothing is ever pruned or rewritten. The report
step rescans all of history/ (retention: forever) and writes a
self-contained static site/ (one page per target + an index; inline SVG
charts, no JS, no server).

Exit codes (probe): 0 = every target produced a report (losses are data,
not errors), 1 = at least one llmping invocation produced no report at all.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PROBES = HERE.parent / "probes.yaml"
DEFAULT_HISTORY = HERE.parent / "history"
DEFAULT_SITE = HERE.parent / "site"
# providers.yaml stays llmping's business: endpoints/auth are discovered by
# the llmping subprocess itself (--providers/$LLMPING_PROVIDERS/./providers.yaml/
# ~/.config/llmping) — this wrapper never reads it.


# --------------------------------------------------------------------- shared


def _load_targets(path: pathlib.Path, include: list[str] | None) -> list[dict[str, Any]]:
    if not path.is_file():
        sys.exit(f"probes file not found: {path} (see scripts/README.md)")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    targets = raw.get("targets") or []
    if not targets:
        sys.exit(f"{path}: no 'targets:' entries")
    seen: set[str] = set()
    for t in targets:
        name = t.get("name")
        if not name:
            sys.exit(f"{path}: every target needs a 'name'")
        if name in seen:
            sys.exit(f"{path}: duplicate target name {name!r}")
        seen.add(name)
        if not t.get("provider") or not t.get("model"):
            sys.exit(f"{path}: target {name!r} needs 'provider' and 'model'")
    if include:
        known = {t["name"] for t in targets}
        unknown = [n for n in include if n not in known]
        if unknown:
            sys.exit(f"unknown target(s) in --only: {', '.join(unknown)}")
        targets = [t for t in targets if t["name"] in include]
    return targets


def _load_groups(path: pathlib.Path) -> dict[str, list[str]]:
    """group name → member target names (in probes.yaml order). Only the
    report uses groups (a comparison page per group); probing ignores them."""
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    groups: dict[str, list[str]] = {}
    for t in raw.get("targets") or []:
        group = t.get("group")
        if group:
            groups.setdefault(str(group), []).append(str(t["name"]))
    return groups


def _llmping_bin() -> str:
    """Find the llmping entrypoint: $LLMPING_BIN, else the venv this script
    runs from (if it also has llmping), else PATH. Keeps the wrapper working
    under cron (bare PATH) and under pipx."""
    env = os.environ.get("LLMPING_BIN")
    if env:
        return env
    exe_dir = pathlib.Path(sys.executable).parent
    if (exe_dir / "llmping").exists():
        return str(exe_dir / "llmping")
    return "llmping"


def _llmping_argv(target: dict[str, Any], providers: str | None) -> list[str]:
    """The llmping command for one target. Secrets never appear here —
    endpoints/auth come from llmping's own providers.yaml discovery."""
    argv = [
        _llmping_bin(),
        "--format",
        "json",
        "--provider",
        str(target["provider"]),
        "--model",
        str(target["model"]),
    ]
    if providers:
        argv += ["--providers", providers]
    if target.get("scenario"):
        argv += ["--scenario", str(target["scenario"])]
    if target.get("count"):
        argv += ["--count", str(target["count"])]
    if target.get("interval"):
        argv += ["--interval", str(target["interval"])]
    if target.get("timeout"):
        argv += ["--timeout", str(target["timeout"])]
    if target.get("extra_body"):
        argv += ["--extra-body", json.dumps(target["extra_body"])]
    return argv


# --------------------------------------------------------------------- probe


def _probe_timeout(target: dict[str, Any]) -> float:
    """Worst-case wall time for one target: count rounds × (interval +
    per-request timeout margin) plus slack. Only a safety net — llmping
    honors --timeout itself; this just stops a hung process from holding
    the whole cron tick hostage."""
    count = int(target.get("count") or 1)
    interval = float(target.get("interval") or 1.0)
    timeout = float(target.get("timeout") or 30.0)
    return count * (interval + timeout) + 120.0


def _extract_report(stdout: str) -> dict[str, Any] | None:
    """llmping -f json prints exactly one JSON object (status goes to
    stderr). Anything else means the probe didn't produce a report."""
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _probe_one(
    target: dict[str, Any], providers: str | None, history_dir: pathlib.Path
) -> dict[str, Any]:
    """Run llmping for one target, append the envelope to today's JSONL."""
    try:
        proc = subprocess.run(
            _llmping_argv(target, providers),
            capture_output=True,
            text=True,
            timeout=_probe_timeout(target),
        )
        envelope = {
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
            "target": target["name"],
            "exit": proc.returncode,
            "report": _extract_report(proc.stdout),
            "stderr": proc.stderr.strip()[-2000:] or None,
        }
    except subprocess.TimeoutExpired:
        envelope = {
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
            "target": target["name"],
            "exit": 124,
            "report": None,
            "stderr": "probe-level timeout expired (llmping hung?)",
        }
    out = history_dir / f"{envelope['ts'][:10]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    return envelope


def cmd_probe(args: argparse.Namespace) -> int:
    targets = _load_targets(pathlib.Path(args.probes), args.only)
    history_dir = pathlib.Path(args.history)
    providers = getattr(args, "providers", None)
    print(f"probing {len(targets)} target(s) → {history_dir}/")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        envelopes = list(pool.map(lambda t: _probe_one(t, providers, history_dir), targets))
    no_report = 0
    for env in envelopes:
        rep = env["report"]
        if rep is None:
            no_report += 1
            print(f"  {env['target']}: NO REPORT (exit {env['exit']})")
        else:
            st = rep.get("stats") or {}
            e2e = (st.get("e2e") or {}).get("avg")
            print(f"  {env['target']}: loss={st.get('loss_pct')}% e2e={e2e}ms exit={env['exit']}")
    print(f"done: {len(envelopes)} run(s), {no_report} without a report")
    if args.report:
        args.days = getattr(args, "days", None)  # probe has no --days of its own
        cmd_report(args)
    return 1 if no_report else 0


# --------------------------------------------------------------------- report


def _load_history(history_dir: pathlib.Path, days: int | None) -> list[dict[str, Any]]:
    """Read history/*.jsonl in chronological order. `days` optionally limits
    the scan window (None = forever)."""
    rows: list[dict[str, Any]] = []
    files = sorted(history_dir.glob("*.jsonl"))
    if days is not None:
        cutoff = (dt.date.today() - dt.timedelta(days=days - 1)).isoformat()
        files = [p for p in files if p.stem >= cutoff]
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn line never takes the site down
    return rows


def _by_target(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:  # files scanned in order → runs are chronological
        out.setdefault(r.get("target", "?"), []).append(r)
    return out


def _series(runs: list[dict[str, Any]], key: tuple[str, ...]) -> list[tuple[str, float]]:
    """(ts, value) pairs for a stat path like ('report','stats','e2e','avg'),
    skipping runs without that value."""
    pts: list[tuple[str, float]] = []
    for r in runs:
        node: Any = r
        for k in key:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        if isinstance(node, (int, float)):
            pts.append((r["ts"], float(node)))
    return pts


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "target"


def _fmt_ms(v: float | None) -> str:
    return f"{v:.0f} ms" if isinstance(v, (int, float)) else "?"


_UPLOT_CSS_URL = "https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.min.css"
_UPLOT_JS_URL = "https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.js"

# One chart spec per metric: (div id, label, series color, stat path)
_CHART_KEYS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("loss", "loss %", "#d33", ("report", "stats", "loss_pct")),
    ("e2e", "e2e avg (ms)", "#4a9ec4", ("report", "stats", "e2e", "avg")),
    ("ttft", "ttft avg (ms)", "#8a6", ("report", "stats", "ttft", "avg")),
]

_CHART_JS = """
function renderCharts(spec) {
  var WINDOWS = [["24h", 86400], ["7d", 7 * 86400], ["30d", 30 * 86400], ["all", null]];
  var DEFAULT_WINDOW = 86400;  // open on the last 24h; the full history stays embedded

  document.querySelectorAll(".chart[data-metric]").forEach(function (div) {
    var metric = div.dataset.metric;
    var data = spec.series[metric];
    if (!data || !data.t.length) {
      div.innerHTML = "<p class='muted'>no data</p>";
      return;
    }
    var fmt = spec.fmt[metric];  // "ms" | "pct"
    // axis.values is (self, ticks[]) => string[] — the whole tick array in,
    // the formatted array out (NOT a per-value formatter).
    var yValues = fmt === "pct"
      ? function (self, ticks) { return ticks.map(function (v) { return v.toFixed(1) + "%"; }); }
      : function (self, ticks) { return ticks.map(function (v) { return v.toFixed(0) + " ms"; }); };

    var t = data.t;
    var tMin = t[0], tMax = t[t.length - 1];

    var u = new uPlot({
      width: Math.min(div.clientWidth || 800, 900),
      height: 240,
      title: spec.titles[metric],
      id: metric,
      legend: { live: true },
      scales: { x: { time: true } },
      series: [
        {},
        { stroke: spec.colors[metric], fill: "rgba(74,158,196,0.08)",
          width: 2, points: { show: false } }
      ],
      axes: [
        {},
        {
          values: yValues,
          // uPlot's default y-axis gutter is a FIXED 50px — it never
          // auto-measures, so '1600 ms' / '100.0%' tick labels get clipped.
          // size (self, values, i) → px; measure the widest formatted label
          // with some padding (canvas 2D metrics, the same font as uPlot's).
          size: function (self, values, i) {
            if (values == null) return 70;  // uPlot probes with null first
            var ctx = document.createElement("canvas").getContext("2d");
            ctx.font = '12px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
            var w = 0;
            for (var k = 0; k < values.length; k++) {
              w = Math.max(w, ctx.measureText(values[k]).width);
            }
            return Math.ceil(w) + 16;  // labelGap + breathing room
          },
        },
      ],
      cursor: { drag: { x: true, y: true } }
    }, [t, data.v], div);

    // time-window chips: setScale("x", {min,max}) — the full dataset stays
    // in memory, so switching windows is instant (no refetch, static page)
    var bar = document.createElement("div");
    bar.className = "winbar";
    WINDOWS.forEach(function (w) {
      var b = document.createElement("button");
      b.textContent = w[0];
      b.onclick = function () {
        // window = the last w[1] seconds ending at the newest point, clamped
        // to the available span; null = everything
        var min = w[1] === null ? tMin : Math.max(tMin, tMax - w[1]);
        u.setScale("x", { min: min, max: tMax });
        Array.prototype.forEach.call(bar.children, function (c) { c.classList.remove("active"); });
        b.classList.add("active");
      };
      bar.appendChild(b);
    });
    div.parentNode.insertBefore(bar, div);
    // default view: last 24h (or everything when history is shorter)
    Array.prototype.forEach.call(bar.children, function (c) {
      if (c.textContent === "24h") c.click();
    });
  });
}

// Group pages: one chart per metric, one line per member target, shared
// wall-clock x axis. Same window chips as the single-target charts.
function renderMultiCharts(spec) {
  var WINDOWS = [["24h", 86400], ["7d", 7 * 86400], ["30d", 30 * 86400], ["all", null]];
  document.querySelectorAll(".mchart[data-metric]").forEach(function (div) {
    var metric = div.dataset.metric;
    var members = spec.series[metric];       // name → {t, v}
    var names = Object.keys(members).filter(function (n) { return members[n].t.length; });
    if (!names.length) {
      div.innerHTML = "<p class='muted'>no data</p>";
      return;
    }
    // Bucket to the minute (floor, not nearest: nearby runs must land in the
    // SAME bucket — nearest-rounding can split runs 20 s apart into adjacent
    // buckets). Runs within a bucket aggregate to their mean; a member with
    // no run in a bucket gets a null → a genuine gap in its line. This both
    // aligns members probed seconds apart and avoids the union axis turning
    // every line into null-broken segments.
    var BUCKET = 60;
    function bucket(t) { return Math.floor(t / BUCKET) * BUCKET; }
    // per member: bucket → mean of that member's values in the bucket
    var perMember = names.map(function (n) {
      var sums = {}, counts = {};
      members[n].t.forEach(function (t, i) {
        var b = bucket(t);
        sums[b] = (sums[b] || 0) + members[n].v[i];
        counts[b] = (counts[b] || 0) + 1;
      });
      var out = {};
      Object.keys(sums).forEach(function (b) { out[b] = sums[b] / counts[b]; });
      return out;
    });
    var allT = [];  // sorted union of member buckets
    perMember.forEach(function (m) {
      Object.keys(m).forEach(function (b) { allT.push(Number(b)); });
    });
    allT = allT.sort(function (a, b) { return a - b; }).filter(function (v, i, arr) {
      return i === 0 || v !== arr[i - 1];
    });
    var aligned = names.map(function (n, i) {
      return allT.map(function (b) {
        return Object.prototype.hasOwnProperty.call(perMember[i], b)
          ? perMember[i][b] : null;
      });
    });
    var fmt = spec.fmt[metric];
    var yValues = fmt === "pct"
      ? function (self, ticks) { return ticks.map(function (v) { return v.toFixed(1) + "%"; }); }
      : function (self, ticks) { return ticks.map(function (v) { return v.toFixed(0) + " ms"; }); };
    var series = [{}, ...names.map(function (n, i) {
      return { label: n, stroke: spec.palette[i % spec.palette.length],
               width: 2, points: { show: false },
               spanGaps: true };  // bridge buckets the member skipped — a
                                  // missed round isn't a line break
    })];
    var tMin = allT[0], tMax = allT[allT.length - 1];
    var u = new uPlot({
      width: Math.min(div.clientWidth || 900, 1000),
      height: 280,
      title: spec.titles[metric],
      id: "m-" + metric,
      legend: { live: true },
      scales: { x: { time: true } },
      series: series,
      axes: [
        {},
        {
          values: yValues,
          size: function (self, values, i) {
            if (values == null) return 70;  // uPlot probes with null first
            var ctx = document.createElement("canvas").getContext("2d");
            ctx.font = '12px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
            var w = 0;
            for (var k = 0; k < values.length; k++) {
              w = Math.max(w, ctx.measureText(values[k]).width);
            }
            return Math.ceil(w) + 16;
          },
        },
      ],
      cursor: { drag: { x: true, y: true } }
    }, [allT, ...aligned], div);
    var bar = document.createElement("div");
    bar.className = "winbar";
    WINDOWS.forEach(function (w) {
      var b = document.createElement("button");
      b.textContent = w[0];
      b.onclick = function () {
        var min = w[1] === null ? tMin : Math.max(tMin, tMax - w[1]);
        u.setScale("x", { min: min, max: tMax });
        Array.prototype.forEach.call(bar.children, function (c) { c.classList.remove("active"); });
        b.classList.add("active");
      };
      bar.appendChild(b);
    });
    div.parentNode.insertBefore(bar, div);
    Array.prototype.forEach.call(bar.children, function (c) {
      if (c.textContent === "24h") c.click();
    });
  });
}
"""


def _chart_data_js(runs: list[dict[str, Any]]) -> str:
    """One JSON blob per target page: {t: epoch seconds, v: values} per metric,
    rendered client-side by uPlot (CDN)."""
    series: dict[str, dict[str, list[float]]] = {}
    colors: dict[str, str] = {}
    titles: dict[str, str] = {}
    for cid, label, color, key in _CHART_KEYS:
        pts = _series(runs, key)
        series[cid] = {
            "t": [_ts_to_epoch(ts) for ts, _ in pts],
            "v": [v for _, v in pts],
        }
        colors[cid] = color
        titles[cid] = label
    blob = json.dumps(
        {
            "series": series,
            "colors": colors,
            "titles": titles,
            "fmt": {"loss": "pct", "e2e": "ms", "ttft": "ms"},
        }
    )
    return f"renderCharts({blob});"


def _ts_to_epoch(ts: str) -> float:
    """ISO-8601 envelope ts (UTC) → epoch seconds for uPlot's time axis."""
    try:
        return (
            dt.datetime.fromisoformat(ts).timestamp()
            if "+" in ts or "Z" in ts
            else dt.datetime.fromisoformat(ts).replace(tzinfo=dt.UTC).timestamp()
        )
    except ValueError:
        return 0.0


def _sparkline(pts: list[tuple[str, float]], color: str, w: int = 240, h: int = 40) -> str:
    """Tiny inline-SVG thumbnail (index cards). The per-target pages use
    uPlot; this keeps the index dependency-free and instant to scan."""
    if not pts:
        return ""
    pad = 3
    vals = [v for _, v in pts]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = (w - 2 * pad) / max(len(pts) - 1, 1)
    d = " ".join(
        f"{pad + i * step:.1f},{h - pad - (v - lo) / span * (h - 2 * pad):.1f}"
        for i, (_, v) in enumerate(pts)
    )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{d}"/>'
        "</svg>"
    )


def _target_status(runs: list[dict[str, Any]]) -> tuple[str, str]:
    """(css class, label) from the newest run."""
    if not runs:
        return "unknown", "no data"
    last = runs[-1]
    rep = last.get("report")
    if rep is None:
        return "down", f"no report (exit {last.get('exit')})"
    loss = float((rep.get("stats") or {}).get("loss_pct") or 0.0)
    if loss == 0.0:
        return "ok", "0% loss"
    if loss < 100.0:
        return "degraded", f"{loss:.0f}% loss"
    return "down", f"{loss:.0f}% loss"


_CSS = """
 body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
        padding: 0 1rem; color: #222; background: #fff }
 h1 { font-size: 1.35rem; margin-bottom: .2rem } h2 { font-size: 1.05rem; margin-top: 1.6rem }
 a { color: #15c; text-decoration: none } .muted { color: #888 }
 .mono { font-family: ui-monospace, monospace; font-size: .85rem }
 table { border-collapse: collapse; width: 100%; margin-top: .5rem }
 th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #eee }
 th { color: #666; font-weight: 600; font-size: .8rem }
 .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
        vertical-align: middle; margin-right: .4rem }
 .ok { background: #2c8 } .degraded { background: #e9a13b }
 .down { background: #d33 } .unknown { background: #999 }
 .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr));
          gap: 1rem; margin-top: 1rem }
 .card { border: 1px solid #e2e2e2; border-radius: 8px; padding: .8rem 1rem }
 .card:hover { border-color: #bbb }
 .card .name { font-weight: 600 }
 .nums { display: flex; gap: 1.4rem; margin-top: .5rem; font-size: .9rem }
 .nums div span { display: block; color: #888; font-size: .72rem }
 pre { background: #f6f6f6; padding: .6rem; border-radius: 6px; overflow-x: auto }
 .winbar { display: flex; gap: .4rem; margin-top: 1.2rem }
 .winbar button { border: 1px solid #ddd; background: #fafafa; border-radius: 6px;
                  padding: .15rem .7rem; font-size: .8rem; color: #555; cursor: pointer }
 .winbar button:hover { border-color: #bbb }
 .winbar button.active { background: #4a9ec4; border-color: #4a9ec4; color: #fff }
"""


def _page(title: str, body: str, css: str = _CSS, js_data: str | None = None) -> str:
    """js_data: a renderer call with embedded JSON (renderCharts for single
    target pages, renderMultiCharts for group pages) — when given, the page
    loads uPlot from a CDN; without it, plain HTML."""
    head_extra = ""
    tail_extra = ""
    if js_data is not None:
        head_extra = (
            f'<link rel="stylesheet" href="{_UPLOT_CSS_URL}">\n'
            f'<script src="{_UPLOT_JS_URL}"></script>\n'
            f"<script>{_CHART_JS}</script>"
        )
        tail_extra = f"<script>{js_data}</script>"
    generated = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    footer = (
        f"generated {generated} · charts by uPlot (CDN)" if js_data else f"generated {generated}"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{head_extra}
<style>{css}</style></head><body>
{body}
{tail_extra}
<p class="muted">{footer}</p>
</body></html>"""


def cmd_report(args: argparse.Namespace) -> int:
    history_dir = pathlib.Path(args.history)
    site_dir = pathlib.Path(args.site)
    rows = _load_history(history_dir, args.days)
    by_target = _by_target(rows)
    if not by_target:
        print(f"no history in {history_dir}/ — run 'probe' first")
        return 1
    site_dir.mkdir(parents=True, exist_ok=True)
    probes_path = pathlib.Path(getattr(args, "probes", "") or DEFAULT_PROBES)
    groups = _load_groups(probes_path)
    if not groups and not probes_path.is_file():
        # not an error — but if the user *has* groups configured elsewhere,
        # this is why no comparison pages appear
        print(f"note: {probes_path} not found — no group pages (pass --probes?)")
    # groups the report will actually render (≥2 members with data)
    groups_layout = [
        g for g, members in groups.items() if sum(1 for m in members if m in by_target) >= 2
    ]
    skipped = [g for g in groups if g not in groups_layout]
    if skipped:
        print("note: group(s) skipped (fewer than 2 members have history): " + ", ".join(skipped))

    def chart_div(cid: str) -> str:
        """Placeholder div; renderCharts() fills it from the embedded JSON."""
        return f'<div class="chart" data-metric="{cid}" style="margin-top:.5rem"></div>'

    # index -------------------------------------------------------------- #
    cards: list[str] = []
    for name in sorted(by_target):
        runs = by_target[name]
        status, label = _target_status(runs)
        loss_pts = _series(runs, ("report", "stats", "loss_pct"))
        e2e_pts = _series(runs, ("report", "stats", "e2e", "avg"))
        ttft_pts = _series(runs, ("report", "stats", "ttft", "avg"))
        cards.append(
            f'<div class="card"><a href="{_slug(name)}.html">'
            f'<span class="dot {status}"></span><span class="name">{html.escape(name)}</span>'
            f'<div class="muted">{html.escape(label)} · {len(runs)} runs</div>'
            '<div class="nums">'
            f"<div><span>loss last</span>{loss_pts[-1][1]:.0f}%</div>"
            f"<div><span>e2e avg</span>{_fmt_ms(e2e_pts[-1][1] if e2e_pts else None)}</div>"
            f"<div><span>ttft avg</span>{_fmt_ms(ttft_pts[-1][1] if ttft_pts else None)}</div>"
            "</div>" + _sparkline(e2e_pts, "#4a9ec4", w=240, h=32) + "</a></div>"
        )
    index_body = (
        "<h1>llmping</h1><p class='muted'>recurring probes · "
        f"{len(rows)} runs across {len(by_target)} target(s)</p>"
        + (
            "<p class='muted'>groups: "
            + " · ".join(
                f'<a href="group-{_slug(g)}.html">{html.escape(g)}</a>' for g in groups_layout
            )
            + "</p>"
            if groups_layout
            else ""
        )
        + '<div class="cards">'
        + "".join(cards)
        + "</div>"
    )
    (site_dir / "index.html").write_text(_page("llmping", index_body), encoding="utf-8")

    # per-target pages ---------------------------------------------------- #
    for name, runs in by_target.items():
        status, label = _target_status(runs)
        head = (
            f'<h1><span class="dot {status}"></span>{html.escape(name)}</h1>'
            f"<p class='muted'>{html.escape(label)} · provider "
            f"{html.escape(str((runs[-1].get('report') or {}).get('provider', '?')))} · model "
            f"{html.escape(str((runs[-1].get('report') or {}).get('model', '?')))} · "
            f"{len(runs)} runs</p>"
        )
        charts = (
            "<h2>loss %</h2>"
            + chart_div("loss")
            + "<h2>e2e avg (ms)</h2>"
            + chart_div("e2e")
            + "<h2>ttft avg (ms)</h2>"
            + chart_div("ttft")
        )
        last_rows = runs[-10:]
        tbl = (
            "<h2>latest runs</h2><table>"
            "<tr><th>when</th><th>loss</th><th>e2e avg</th><th>ttft avg</th><th>exit</th></tr>"
        )
        for r in reversed(last_rows):
            st = (r.get("report") or {}).get("stats") or {}
            e2e = (st.get("e2e") or {}).get("avg")
            ttft = (st.get("ttft") or {}).get("avg")
            tbl += (
                f"<tr><td class='mono'>{html.escape(r['ts'][:19])}</td>"
                f"<td>{st.get('loss_pct', '?')}%</td>"
                f"<td>{_fmt_ms(e2e)}</td><td>{_fmt_ms(ttft)}</td>"
                f"<td>{r.get('exit')}</td></tr>"
            )
        tbl += "</table>"
        if runs[-1].get("stderr"):
            tbl += "<h2>last stderr</h2><pre>" + html.escape(str(runs[-1]["stderr"])) + "</pre>"
        (site_dir / f"{_slug(name)}.html").write_text(
            _page(name, head + charts + tbl, js_data=_chart_data_js(runs)), encoding="utf-8"
        )

    # group pages ---------------------------------------------------------- #
    n_group_pages = 0
    for gname in groups_layout:
        members = groups[gname]
        member_runs = {m: by_target[m] for m in members if m in by_target}
        if len(member_runs) < 2:
            continue  # a comparison page needs at least two targets with data
        n_group_pages += 1
        series: dict[str, dict[str, dict[str, list[Any]]]] = {}
        titles: dict[str, str] = {}
        for cid, label, _color, key in _CHART_KEYS:
            series[cid] = {}
            for m, mruns in member_runs.items():
                pts = _series(mruns, key)
                series[cid][m] = {
                    "t": [_ts_to_epoch(ts) for ts, _ in pts],
                    "v": [v for _, v in pts],
                }
            titles[cid] = label
        blob = json.dumps(
            {
                "series": series,
                "titles": titles,
                "palette": ["#4a9ec4", "#d33", "#8a6", "#a6c", "#e93", "#57b"],
                "fmt": {"loss": "pct", "e2e": "ms", "ttft": "ms"},
            }
        )
        roster = " · ".join(f'<a href="{_slug(m)}.html">{html.escape(m)}</a>' for m in member_runs)
        body = (
            f"<h1>{html.escape(gname)}</h1>"
            f"<p class='muted'>comparison across {len(member_runs)} targets — {roster}</p>"
            + "<h2>loss %</h2><div class='mchart' data-metric='loss'></div>"
            + "<h2>e2e avg (ms)</h2><div class='mchart' data-metric='e2e'></div>"
            + "<h2>ttft avg (ms)</h2><div class='mchart' data-metric='ttft'></div>"
        )
        (site_dir / f"group-{_slug(gname)}.html").write_text(
            _page(gname, body, js_data=f"renderMultiCharts({blob});"), encoding="utf-8"
        )

    print(
        f"report: {len(by_target)} target page(s)"
        + (f" + {n_group_pages} group page(s)" if n_group_pages else "")
        + f" + index → {site_dir}/"
    )
    return 0


# ----------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # shared by both subcommands (so `probe --report` can reach them too)
    ap.add_argument("--history", default=str(DEFAULT_HISTORY), help="history dir (JSONL per day)")
    ap.add_argument("--site", default=str(DEFAULT_SITE), help="output dir for the HTML site")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="run every target once, append to history/")
    p.add_argument("--probes", default=str(DEFAULT_PROBES), help="path to probes.yaml")
    p.add_argument("--only", action="append", help="probe only this target (repeatable)")
    p.add_argument("--jobs", type=int, default=6, help="parallel llmping processes")
    p.add_argument("--report", action="store_true", help="regenerate the site afterwards")
    p.add_argument(
        "--providers", default=None, help="providers.yaml path passed through to llmping"
    )
    p.set_defaults(func=cmd_probe)

    r = sub.add_parser("report", help="scan history/ and write the static site")
    r.add_argument(
        "--probes", default=str(DEFAULT_PROBES), help="probes.yaml (for 'group:' comparison pages)"
    )
    r.add_argument("--days", type=int, default=None, help="only report the last N days")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
