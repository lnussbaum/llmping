"""llmping CLI. Exit codes: 0 ok · 1 degraded · 2 config/usage
error · 130 Ctrl-C on `--count inf`.

Single command — no subcommand. Usage shape: `llmping [--provider NAME]
--model MODEL [--scenario NAME] [--extra-body JSON]` (exactly one model
per run), or `--list-models` for a pure provider query. Endpoints/auth
come from providers.yaml (with LLMPING_BASE_URL*/LLMPING_TOKEN* env
fallback) — never from CLI flags, so secrets stay out of shell history
and `ps`. What to ping is given on the command line (no config.yaml).
Nothing is written to disk unless `--dump-traffic` is on (the JSON
report is stdout-only).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Never, TypedDict

import typer
from rich.console import Console
from rich.markup import escape

import llmping
from llmping import rundir
from llmping.client import LLMClient, RequestError
from llmping.config import (
    ConfigError,
    ProviderEntry,
    ProvidersConfig,
    Target,
    discover_providers,
    load_providers,
)
from llmping.ping import ModelStats, PingStats, Sample, run_ping
from llmping.scenario import DEFAULT_SCENARIO, SCENARIOS, ScenarioError, load_scenario


class PingReport(TypedDict):
    """`--format json` payload (stdout only — nothing is written to disk).
    One run pings a single model, so the stats block is flat, not keyed
    by model name."""

    providers_path: str
    provider: str
    model: str
    scenario: str
    base_url: str
    stats: ModelStats
    duration_s: float
    timestamp: str


class ModelsReport(TypedDict):
    """`--list-models --format json` payload."""

    provider: str
    base_url: str
    models: list[str]


# `quiet` keeps stdout machine-readable in --format json mode. Both consoles
# leave their stream unbound (`stderr=True`, not `file=sys.stderr`) so rich
# resolves sys.stdout/sys.stderr at print time — that's what makes CliRunner
# capture work without rebinding the streams.
console = Console()
err_console = Console(stderr=True)


def _err(msg: str) -> None:
    err_console.print(f"[bold red][ERROR][/] {escape(msg)}")


def _warn(msg: str) -> None:
    err_console.print(f"[yellow][WARN][/] {escape(msg)}")


_COMMANDLINE = "COMMANDLINE"


def _fmt_from_args(args: list[str]) -> str | None:
    """Value of `--format`/`--format=X` from the raw argv, for paths that
    run before click parses anything (the bare `--scenario` listing)."""
    it = iter(args)
    for a in it:
        if a == "--format":
            return next(it, None)
        if a.startswith("--format="):
            return a.split("=", 1)[1]
    return None


def _explicit(ctx: typer.Context, name: str) -> bool:
    """True when the option was actually given on the command line.

    typer vendors click (``typer._click``), so ``click.core.ParameterSource``
    is a *different* enum from the one ``get_parameter_source`` returns, and
    importing the vendored module is private API — compare the member name.
    """
    source = ctx.get_parameter_source(name)
    return source is not None and source.name == _COMMANDLINE


def _bare_scenario(args: list[str]) -> bool:
    """True when `--scenario`/`-s` appears without a value: trailing, or
    followed by another flag (`--scenario --count 2`). The `--scenario=X`
    form always carries its value and never matches."""
    for i, a in enumerate(args):
        if a in ("--scenario", "-s"):
            nxt = args[i + 1] if i + 1 < len(args) else None
            if nxt is None or nxt.startswith("-"):
                return True
    return False


def _flag_in_args(args: list[str], flag: str) -> bool:
    """True when `flag` appears in raw argv, as `--flag` or `--flag=…`."""
    return flag in args or any(a.startswith(flag + "=") for a in args)


# Flags the bare `--scenario` listing ignores (warned about, exactly like
# --list-models warns on the flags IT ignores). Scenarios are code-defined
# and global, so even --provider/--providers have no effect on the listing.
_SCENARIO_LISTING_IGNORES = (
    "--providers",
    "--provider",
    "-p",
    "--extra-body",
    "--count",
    "-c",
    "--interval",
    "-i",
    "--timeout",
    "-t",
    "--runs-dir",
    "--dump-traffic",
    "-d",
)


class _ScenarioListGroup(typer.core.TyperGroup):
    """Root group that answers a **bare `--scenario`** (no value) with the
    list of valid scenario names, exit 0.

    typer 0.29 vendors a stripped click whose parser rejects a valueless
    option ("requires an argument") before any callback runs, and its
    `is_flag`/`flag_value` pair is deprecated and does not give click's
    optional-value semantics — so the only hook that sees `--scenario`
    before the parser does is `parse_args`.

    Like `--help`, a bare `--scenario` answers regardless of what else is
    on the command line (scenarios are code-defined and global — nothing
    else can change the answer); flags that would have shaped a run are
    warned about, not errors.
    """

    def parse_args(self, ctx: typer._click.core.Context, args: list[str]) -> list[str]:
        if _bare_scenario(args) and not ctx.resilient_parsing:
            # `--format` isn't parsed yet at this point, so honour it the
            # same way the callback would (strict values; JSON on stdout).
            fmt = _fmt_from_args(args)
            if fmt is not None and fmt not in {"table", "json"}:
                _err(f"--format must be 'table' or 'json', got {fmt!r}")
                _exit(2)  # never returns
            ignored = [f for f in _SCENARIO_LISTING_IGNORES if _flag_in_args(args, f)]
            if _flag_in_args(args, "--model"):
                ignored = ["--model", *ignored]
            if ignored:
                _warn(f"--scenario with no value pings nothing — ignoring {', '.join(ignored)}")
            if fmt == "json":
                print(json.dumps({"scenarios": list(SCENARIOS)}))
                _exit(0)  # never returns
            # The callback that manages console.quiet never runs on this
            # path — make the listing self-contained (a stale quiet=True
            # from an earlier callback run in this process must not
            # swallow the table).
            console.quiet = False
            console.print("scenarios: " + ", ".join(SCENARIOS))
            _exit(0)  # never returns
        return super().parse_args(ctx, args)


app = typer.Typer(
    name="llmping",
    help=(
        "Ping OpenAI-compatible LLM servers.\n\n"
        "Providers (endpoints + tokens) come from providers.yaml, or from "
        "the environment: $LLMPING_PROVIDER (default --provider), "
        "$LLMPING_BASE_URL / $LLMPING_TOKEN (fill in any provider that "
        "omits them; per-provider $LLMPING_BASE_URL_<NAME> / "
        "$LLMPING_TOKEN_<NAME> take precedence), and $LLMPING_PROVIDERS "
        "(path to the providers file)."
    ),
    no_args_is_help=True,
    invoke_without_command=True,
    add_completion=False,  # no --install-completion/--show-completion noise
    context_settings={"help_option_names": ["-h", "--help"]},
    cls=_ScenarioListGroup,
)


def _exit(code: int) -> Never:
    raise typer.Exit(code)


# --------------------------------------------------------------------------- helpers


def _load_providers(explicit: str | None) -> tuple[ProvidersConfig, Path]:
    try:
        path = discover_providers(explicit)
        providers = load_providers(path)
    except ConfigError as e:
        _err(str(e))
        _exit(2)  # never returns
    return providers, path


def _resolve_count(value: str | None) -> int | None:
    """None = run until interrupted (`--count inf`)."""
    if value is None:
        return 1
    v = value.strip().lower()
    if v in {"inf", "infinity", "unlimited"}:
        return None
    try:
        n = int(v)
    except ValueError:
        _err("--count must be an int >= 1 or 'inf'")
        _exit(2)  # never returns
    if n < 1:
        _err("--count must be an int >= 1 or 'inf'")
        _exit(2)  # never returns
    return n


def _parse_extra_body(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        _err(f"--extra-body must be valid JSON: {e}")
        _exit(2)  # never returns
    if not isinstance(parsed, dict):
        _err("--extra-body must be a JSON object")
        _exit(2)  # never returns
    return parsed


def _client(
    prov: ProviderEntry,
    timeout: float,
    dump_exchange: Callable[[dict[str, Any]], None] | None = None,
) -> LLMClient:
    # Connection (base_url + token) comes from the provider entry only —
    # providers.yaml or the LLMPING_BASE_URL*/LLMPING_TOKEN* env vars.
    # There are deliberately no --base-url/--token flags: secrets on a
    # command line leak into shell history and ps.
    return LLMClient(prov.base_url, prov.token, timeout=timeout, dump_exchange=dump_exchange)


def _list_models(
    prov: ProviderEntry,
    timeout: float,
    fmt: str,
) -> Never:
    """`llmping --list-models`: GET {base_url}/models, print the advertised
    model IDs, exit 0. A failed request (timeout / transport / HTTP error)
    prints the error and exits 1 — the provider did not answer, like a lost
    ping. Pure query: no run dir, no report, nothing pinged.
    """
    client = _client(prov, timeout)

    async def _fetch() -> list[str]:
        try:
            return await client.models()
        finally:
            await client.close()

    try:
        names = asyncio.run(_fetch())
    except RequestError as exc:
        _err(str(exc))
        _exit(1)  # never returns
    if fmt == "json":
        payload: ModelsReport = {
            "provider": prov.name,
            "base_url": client.base_url,
            "models": names,
        }
        print(json.dumps(payload, indent=2))
    else:
        console.print(f"models on {escape(prov.name)} ({escape(client.base_url)}): {len(names)}")
        for name in names:
            console.print(f"  {escape(name)}")
    _exit(0)  # never returns


def _traffic_dump(dump_traffic: bool, runs_dir: Path, run_dir: rundir.RunDir) -> rundir.TrafficDump:
    """--dump-traffic capture sink → run dir traffic-<seq>.jsonl."""
    if not dump_traffic:
        return rundir.TrafficDump.disabled()
    assert run_dir.path is not None  # RunDir is created iff dump_traffic
    return rundir.TrafficDump(run_dir.path / "traffic.jsonl")


# --------------------------------------------------------------------------- cli


def _version_callback(value: bool) -> None:
    if value:
        print(f"llmping {llmping.__version__}")
        raise typer.Exit(0)


@app.callback()
def cli(
    ctx: typer.Context,
    # --- target: what to ping ------------------------------------------- #
    provider: str = typer.Option(
        None,
        "-p",
        "--provider",
        envvar="LLMPING_PROVIDER",
        help=(
            "Provider name from providers.yaml "
            "(default: $LLMPING_PROVIDER; when unset, a provider named "
            "'default' in providers.yaml)"
        ),
    ),
    model: list[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="Bare API model name to ping (exactly one per run)",
    ),
    scenario: str = typer.Option(
        DEFAULT_SCENARIO,
        "-s",
        "--scenario",
        help=(
            "Code-defined scenario to run (default: pingpong, the plain probe); "
            "pass --scenario with no value to list the valid names"
        ),
    ),
    # --- run shape: how long, how fast ---------------------------------- #
    count: str | None = typer.Option(
        None,
        "-c",
        "--count",
        help="Rounds; 'inf' (or 'infinity'/'unlimited') = until Ctrl-C (default: 1)",
    ),
    interval: float = typer.Option(1.0, "-i", "--interval", help="Seconds between round starts"),
    timeout: float = typer.Option(30.0, "-t", "--timeout", help="Per-request timeout (s)"),
    extra_body: str = typer.Option(
        None,
        "--extra-body",
        help="Extra request body as a JSON object, e.g. '{\"temperature\": 0}'",
    ),
    # --- output ---------------------------------------------------------- #
    fmt: str = typer.Option("table", "-f", "--format", help="table | json"),
    dump_traffic: bool = typer.Option(
        False,
        "-d",
        "--dump-traffic",
        help="Log every HTTP exchange (raw + decoded) to <run dir>/traffic-<seq>.jsonl",
    ),
    runs_dir: str = typer.Option("./runs", "--runs-dir", help="Where run dirs live"),
    # --- config & meta --------------------------------------------------- #
    providers: str = typer.Option(
        None,
        "--providers",
        envvar="LLMPING_PROVIDERS",
        help=(
            "Path to providers.yaml (default: ./providers.yaml or ~/.config/llmping/providers.yaml)"
        ),
    ),
    list_models: bool = typer.Option(
        False,
        "-l",
        "--list-models",
        help="List the models the provider advertises (GET /models) and exit",
    ),
    version: bool = typer.Option(
        None,
        "-V",
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the llmping version and exit",
    ),
) -> None:
    if fmt not in {"table", "json"}:
        # strict like every other flag: an unknown --format is a usage error,
        # not a silent fallback that makes monitoring output look table-ish
        _err(f"--format must be 'table' or 'json', got {fmt!r}")
        _exit(2)  # never returns
    console.quiet = fmt == "json"

    config, providers_path = _load_providers(providers)
    if provider is None:  # neither --provider nor $LLMPING_PROVIDER
        provider = "default"
    try:
        prov = config.by_name(provider)
    except ConfigError as exc:
        # name the fallback so the user can see what was tried
        _err(f"{exc} (provider resolution: --provider > $LLMPING_PROVIDER > 'default')")
        _exit(2)  # never returns
    if list_models:
        if model:
            _err("--list-models pings nothing — drop --model (exit 2)")
            _exit(2)  # never returns
        ignored = [
            flag
            for flag, name in (
                ("--scenario", "scenario"),
                ("--extra-body", "extra_body"),
                ("--count", "count"),
                ("--interval", "interval"),
                ("--runs-dir", "runs_dir"),
                ("--dump-traffic", "dump_traffic"),
            )
            if _explicit(ctx, name)
        ]
        if ignored:
            _warn(f"--list-models pings nothing — ignoring {', '.join(ignored)}")
        _list_models(prov, timeout, fmt)  # never returns
    if not model:
        _err(
            "missing --model (or pass --list-models, or a bare --scenario, "
            "for the model-less queries)"
        )
        _exit(2)  # never returns
    if len(model) > 1:
        # One model per run: a repeat is a usage error, not "last one wins".
        _err(f"single --model per run (got {len(model)}: {', '.join(model)})")
        _exit(2)  # never returns
    model_name = model[0]
    try:
        target = Target(
            model=model_name, scenario=scenario, extra_body=_parse_extra_body(extra_body)
        )
    except ConfigError as exc:
        _err(str(exc))
        _exit(2)  # never returns
    try:  # resolve the scenario name up front → exit 2 on unknown
        load_scenario(target.scenario)
    except ScenarioError as exc:
        _err(str(exc))
        _exit(2)

    n_rounds = _resolve_count(count)
    # inf/nan pass a plain `<= 0` check but corrupt the run: `--interval
    # inf` sleeps forever after round 1, `--interval nan` skips the sleep
    # entirely (busy-hammering the server), `--timeout nan` fails every
    # request with httpx's "deadline must not be NaN".
    if not (math.isfinite(interval) and interval > 0):
        _err("--interval must be a finite number > 0")
        _exit(2)  # never returns
    if not (math.isfinite(timeout) and timeout > 0):
        _err("--timeout must be a finite number > 0")
        _exit(2)  # never returns

    # The run dir exists only for --dump-traffic (it holds the exchange
    # logs and nothing else — the report is stdout-only). Named
    # provider_model_scenario_<ts>; `/` in any part is escaped by RunDir.
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_name = f"{prov.name}_{target.model}_{scenario}_{ts}"
    run_dir = rundir.RunDir(Path(runs_dir), run_name) if dump_traffic else rundir.RunDir.none()
    dump = _traffic_dump(dump_traffic, Path(runs_dir), run_dir)
    t0 = time.time()
    started = rundir.now_iso()

    def on_sample(sample: Sample) -> None:
        # Model names and server error text are outside our control — escape
        # them or a bracket-bearing name (e.g. "m[/]") crashes rich with
        # MarkupError instead of printing the line.
        if sample.ok:
            ttft = f" ttft={sample.ttft_ms:.1f}ms" if sample.ttft_ms is not None else ""
            line = (
                f"reply from {escape(sample.model)}: seq={sample.seq}"
                f"{ttft} e2e={sample.e2e_ms:.1f}ms"
            )
            if sample.turns is not None:
                line += f" turns={sample.turns}"
            if sample.output_tokens is not None:
                line += f" tokens={sample.output_tokens}"
            if sample.chars is not None:
                line += f" chars={sample.chars}"
            console.print(line)
        else:
            console.print(
                f"[red]no reply from {escape(sample.model)}: seq={sample.seq} "
                f"({escape(sample.error or '')})[/]"
            )

    client = _client(prov, timeout, dump.record)

    async def _run() -> tuple[list[PingStats], bool]:
        # Close the httpx client inside the very event loop that used it:
        # `asyncio.run` tears that loop down on return, so a later close
        # (or an unclosed client) would leak connections.
        try:
            return await run_ping(
                client,
                [target],
                count=n_rounds,
                interval=interval,
                on_sample=on_sample,
                on_round=lambda s: dump.begin(f"{s:06d}"),
            )
        finally:
            await client.close()

    try:
        run_stats, interrupted = asyncio.run(_run())
    finally:
        # Nothing writes traffic after the run returns, so finalize and
        # release the dump now — before summary/finalize work can raise.
        dump.end()
        dump.close()

    stats = run_stats[0]  # single-model run: exactly one PingStats
    st = stats.stats()
    report: PingReport = {
        "providers_path": str(providers_path),
        "provider": prov.name,
        "model": target.model,
        "scenario": scenario,
        "base_url": prov.base_url,
        "stats": st,
        "duration_s": round(time.time() - t0, 3),
        "timestamp": started,
    }
    console.print("\n" + "--- ping statistics ---")
    console.print(f"provider: {escape(provider)}")
    console.print(f"model: {escape(model_name)}")
    console.print(f"scenario: {escape(scenario)}")
    console.print(f"target: {escape(prov.base_url)}")
    console.print(f"{st['sent']} requests sent, {st['replies']} received, {st['loss_pct']}% loss")
    e2e_stats = st["e2e"]
    if e2e_stats["min"] is not None:
        ems = (
            f"{e2e_stats['min']:.1f}/{e2e_stats['avg']:.1f}"
            f"/{e2e_stats['max']:.1f}/{e2e_stats['mdev']:.1f}"
        )
        console.print(f"e2e  min/avg/max/mdev = {ems} ms")
    ttft_stats = st["ttft"]
    if ttft_stats["min"] is not None:
        tms = (
            f"{ttft_stats['min']:.1f}/{ttft_stats['avg']:.1f}"
            f"/{ttft_stats['max']:.1f}/{ttft_stats['mdev']:.1f}"
        )
        console.print(f"ttft min/avg/max/mdev = {tms} ms")
    if not stats.ok:
        console.print(f"[red]{escape(model_name)}: loss detected → exit 1[/]")

    loss = not stats.ok
    code = 1 if loss else 0
    if interrupted:
        code = 130
    if fmt == "json":
        print(json.dumps(report, indent=2))
    _exit(code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
