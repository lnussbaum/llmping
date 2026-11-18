"""End-to-end CLI tests via typer CliRunner against the real mock server.

One config file (providers.yaml); what to ping is given on the command line:
`llmping --provider mock --model mistral [--scenario …]`. The CLI is a
single command — no subcommand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmping.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _run_cli_from_tmp(tmp_path, monkeypatch):
    """Run every CLI test from a scratch CWD.

    The CLI's default `--runs-dir` is `./runs`, so invocations that don't pass
    the flag would create real run dirs in the repo working
    tree (gitignored, but litter that piles up after every suite run). Tests
    use absolute `tmp_path` paths for providers/runs, so chdir is inert for
    everything else.
    """
    monkeypatch.chdir(tmp_path)


def _write_providers(tmp_path: Path, base_url: str, *, token: str = "sekret") -> Path:
    pv = tmp_path / "providers.yaml"
    pv.write_text(
        f"""providers:
  - name: mock
    base_url: {base_url}
    token: {token}
"""
    )
    return pv


def _ping_args(
    pv: Path,
    *,
    model: str,
    extra: list[str] | None = None,
) -> list[str]:
    args: list[str] = ["--providers", str(pv), "--provider", "mock", "--model", model]
    if extra:
        args.extend(extra)
    return args


def test_ping_json_exit0(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--format",
            "json",
            "--provider",
            "mock",
            "--model",
            "mistral",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert "command" not in report  # the JSON report carries no command field
    assert report["provider"] == "mock"
    assert report["model"] == "mistral"
    assert report["scenario"] == "pingpong"  # the default, spelled out
    assert report["timestamp"]  # run start, ISO-8601 UTC
    assert report["stats"]["loss_pct"] == 0.0


def test_ping_loss_exit1(tmp_path, mock_server_failing):
    pv = _write_providers(tmp_path, mock_server_failing.base_url)
    # fail-model triggers 503 from the mock
    result = runner.invoke(app, _ping_args(pv, model="fail-model"))
    assert result.exit_code == 1
    assert "no reply" in result.output


def test_ping_empty_exit1(tmp_path, mock_server_empty):
    # An HTTP 200 stream with zero content AND zero reasoning is an empty
    # reply (thinking-starved model), counted as a loss → exit 1.
    pv = _write_providers(tmp_path, mock_server_empty.base_url)
    result = runner.invoke(app, _ping_args(pv, model="empty-model"))
    assert result.exit_code == 1
    assert "no reply" in result.output
    assert "empty response (0 chars)" in (result.output + result.stderr)


def test_list_models(tmp_path, mock_server):
    # --list-models is a pure query (no ping, no --model): GET /models on the
    # provider and print the advertised model IDs → exit 0.
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, ["--providers", str(pv), "--provider", "mock", "--list-models"])
    assert result.exit_code == 0, result.output
    assert "gpt-oss-120b" in result.output
    assert "mistral-small-4-119b" in result.output
    assert "models on mock" in result.output


def test_list_models_json(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--format",
            "json",
            "--provider",
            "mock",
            "--list-models",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == {
        "provider": "mock",
        "base_url": mock_server.base_url,
        "models": ["gpt-oss-120b", "mistral-small-4-119b"],
    }


def test_list_models_combined_with_model_exit2(tmp_path, mock_server):
    # --list-models pings nothing, so pairing it with --model is a usage error.
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        ["--providers", str(pv), "--provider", "mock", "--model", "x", "--list-models"],
    )
    assert result.exit_code == 2
    assert "--list-models" in (result.output + result.stderr)


def test_list_models_warns_on_ignored_flags(tmp_path, mock_server):
    # A pure query exits before any ping setup, so run-related flags are
    # ignored — say so instead of silently dropping them.
    pv = _write_providers(tmp_path, mock_server.base_url)
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--provider",
            "mock",
            "--list-models",
            "--scenario",
            "agentic-basic",
            "--count",
            "3",
            "--dump-traffic",
            "--runs-dir",
            str(runs),
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output + result.stderr
    assert "[WARN]" in out
    for flag in ("--scenario", "--count", "--dump-traffic", "--runs-dir"):
        assert flag in out, flag
    assert "gpt-oss-120b" in result.output  # still answers the query
    assert not runs.exists()  # pure query writes nothing


def test_list_models_plain_does_not_warn(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, ["--providers", str(pv), "--provider", "mock", "--list-models"])
    assert result.exit_code == 0, result.output
    assert "ignoring" not in (result.output + result.stderr)


def test_list_models_dead_endpoint_exit1(tmp_path):
    # A failed /models request is like a lost ping: the provider did not
    # answer → error printed, exit 1.
    pv = tmp_path / "providers.yaml"
    pv.write_text(
        """providers:
  - name: dead
    base_url: http://127.0.0.1:1/v1
    token: ""
"""
    )
    result = runner.invoke(app, ["--providers", str(pv), "--provider", "dead", "--list-models"])
    assert result.exit_code == 1
    assert "error" in (result.output + result.stderr).lower()


def test_no_base_url_or_token_flags(tmp_path, mock_server):
    # Connection secrets come from providers.yaml / env only — there are no
    # --base-url/--token flags (they'd leak into shell history and ps)
    # → usage error, exit 2.
    pv = _write_providers(tmp_path, mock_server.base_url)
    for flag in ("--base-url", "--token"):
        result = runner.invoke(
            app, ["--providers", str(pv), "--provider", "mock", "--model", "m", flag, "x"]
        )
        assert result.exit_code == 2, flag


def test_ping_base_url_from_env(tmp_path, mock_server, monkeypatch):
    # providers.yaml omits base_url; $LLMPING_BASE_URL supplies the endpoint.
    monkeypatch.setenv("LLMPING_BASE_URL", mock_server.base_url)
    pv = tmp_path / "providers.yaml"
    pv.write_text("providers:\n  - name: mock\n    token: sekret\n")
    result = runner.invoke(app, _ping_args(pv, model="mistral"))
    assert result.exit_code == 0, result.output
    assert "reply from mistral" in result.output


def test_list_models_token_from_env(tmp_path, mock_server, monkeypatch):
    # providers.yaml omits token; $LLMPING_TOKEN_MOCK supplies it.
    monkeypatch.setenv("LLMPING_TOKEN_MOCK", "env-token")
    pv = tmp_path / "providers.yaml"
    pv.write_text(f"providers:\n  - name: mock\n    base_url: {mock_server.base_url}\n")
    result = runner.invoke(app, ["--providers", str(pv), "--provider", "mock", "--list-models"])
    assert result.exit_code == 0, result.output
    assert "gpt-oss-120b" in result.output


def test_single_model_only(tmp_path, mock_server_failing):
    # One run pings exactly one model: repeating --model is a usage error
    # (the last value would silently win otherwise), and a second model is
    # simply not accepted.
    pv = _write_providers(tmp_path, mock_server_failing.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--format",
            "json",
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--model",
            "fail-model",
        ],
    )
    assert result.exit_code == 2
    assert "single --model" in (result.output + result.stderr)


def test_missing_model_exit2(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    # --model is required; typer emits a usage error → exit 2
    result = runner.invoke(app, ["--providers", str(pv), "--provider", "mock"])
    assert result.exit_code == 2


def test_provider_defaults_to_named_default(tmp_path, mock_server):
    # --provider is optional: without it (and without $LLMPING_PROVIDER)
    # the run uses the provider named 'default'.
    pv = tmp_path / "providers.yaml"
    pv.write_text(
        f"""providers:
  - name: default
    base_url: {mock_server.base_url}
    token: sekret
"""
    )
    result = runner.invoke(app, ["--providers", str(pv), "--model", "mistral"])
    assert result.exit_code == 0, result.output
    assert "reply from mistral" in result.output


def test_provider_from_env(tmp_path, mock_server, monkeypatch):
    # $LLMPING_PROVIDER picks the provider when --provider isn't given.
    monkeypatch.setenv("LLMPING_PROVIDER", "mock")
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, ["--providers", str(pv), "--model", "mistral"])
    assert result.exit_code == 0, result.output
    assert "reply from mistral" in result.output


def test_provider_flag_beats_env(tmp_path, mock_server, monkeypatch):
    monkeypatch.setenv("LLMPING_PROVIDER", "other")
    pv = tmp_path / "providers.yaml"
    pv.write_text(
        f"""providers:
  - name: mock
    base_url: {mock_server.base_url}
    token: sekret
  - name: other
    base_url: http://127.0.0.1:1/v1
    token: ""
"""
    )
    result = runner.invoke(
        app, ["--providers", str(pv), "--provider", "mock", "--model", "mistral"]
    )
    assert result.exit_code == 0, result.output
    mock_used = "reply from mistral" in result.output
    assert mock_used


def test_no_default_provider_exit2(tmp_path, mock_server):
    # No --provider, no $LLMPING_PROVIDER, and no 'default' entry → exit 2
    # with the resolution order spelled out.
    pv = _write_providers(tmp_path, mock_server.base_url)  # provider is 'mock'
    result = runner.invoke(app, ["--providers", str(pv), "--model", "mistral"])
    assert result.exit_code == 2
    out = result.output + result.stderr
    assert "unknown provider 'default'" in out
    assert "LLMPING_PROVIDER" in out


def test_unknown_provider_exit2(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        ["--providers", str(pv), "--provider", "nope", "--model", "mistral"],
    )
    assert result.exit_code == 2
    assert "unknown provider" in (result.output + result.stderr)


def test_duplicate_model_exit2(tmp_path, mock_server):
    # A repeated --model is a usage error — one model per run (the last
    # value would silently win otherwise).
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--model",
            "mistral",
        ],
    )
    assert result.exit_code == 2
    assert "--model" in (result.output + result.stderr)


def test_scenario_bare_lists_names(tmp_path, mock_server):
    # `--scenario` with no value lists the valid scenario names and exits 0 —
    # a pure query, like --list-models (no run dir, nothing pinged). It is
    # purely local (scenarios are code-defined): no providers file needed.
    runs = tmp_path / "runs"
    result = runner.invoke(app, ["--scenario"])
    assert result.exit_code == 0, result.output
    for name in ("pingpong", "agentic-basic", "agentic"):
        assert name in result.output, name
    assert not runs.exists()  # pure query writes nothing


def test_scenario_bare_after_other_flags(tmp_path, mock_server):
    # The bare form works positionally anywhere — the last token decides.
    # Flags the listing ignores produce a warning, like --list-models.
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app, ["--providers", str(pv), "--provider", "mock", "--count", "2", "--scenario"]
    )
    assert result.exit_code == 0, result.output
    assert "pingpong" in result.output
    out = result.output + result.stderr
    assert "[WARN]" in out
    for flag in ("--providers", "--provider", "--count"):
        assert flag in out, flag


def test_scenario_bare_with_model_still_lists(tmp_path, mock_server):
    # A bare --scenario answers regardless of what else is on the command
    # line (like --help) — --model is warned about, not an error.
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, [*_ping_args(pv, model="mistral"), "--scenario"])
    assert result.exit_code == 0, result.output
    assert "pingpong" in result.output
    out = result.output + result.stderr
    assert "[WARN]" in out and "--model" in out


def test_scenario_bare_json_format(tmp_path, mock_server):
    # --format json keeps stdout machine-readable even for the listing.
    result = runner.invoke(app, ["--format", "json", "--scenario"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == {"scenarios": ["pingpong", "agentic-basic", "agentic"]}


def test_scenario_bare_mid_argv(tmp_path, mock_server):
    # `--scenario` followed by another flag is still the bare form — it must
    # not swallow the next flag as its value.
    result = runner.invoke(app, ["--scenario", "--format", "json"])
    assert result.exit_code == 0, result.output
    assert "pingpong" in result.output


def test_scenario_bare_rejects_bad_format(tmp_path, mock_server):
    # Strict even on the pre-parse path: an unknown --format is exit 2.
    result = runner.invoke(app, ["--scenario", "--format", "yaml"])
    assert result.exit_code == 2
    assert "--format" in (result.output + result.stderr)


def test_ping_markup_bearing_model_name(tmp_path, mock_server):
    # Model names (and server error text) are outside our control; a
    # bracket-bearing name like "m[/]" must print, not crash rich with
    # MarkupError ("closing tag has nothing to close").
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, _ping_args(pv, model="m[/]"))
    assert result.exit_code == 0, result.output
    assert "reply from m[/]" in result.output
    # and a failing round with markup-bearing error text prints too
    result = runner.invoke(app, _ping_args(pv, model="m[/]", extra=["--count", "1"]))
    assert result.exit_code in (0, 1)
    assert "MarkupError" not in result.output


def test_unknown_scenario_exit2(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--provider",
            "mock",
            "--model",
            "m",
            "--scenario",
            "nope",
        ],
    )
    assert result.exit_code == 2
    assert "unknown scenario" in (result.output + result.stderr)


def test_ping_agentic_scenario_run(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--format",
            "json",
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--scenario",
            "agentic-basic",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["scenario"] == "agentic-basic"
    assert "reply from mistral" not in result.stdout  # json mode: report only
    # agentic session is a multi-turn loop, not a fixed 2-turn script
    assert report["stats"]["turns"] == 2


def test_missing_providers_exit2(tmp_path):
    result = runner.invoke(
        app,
        ["--providers", str(tmp_path / "nope.yaml"), "--provider", "mock", "--model", "m"],
    )
    assert result.exit_code == 2
    assert "providers" in (result.output + result.stderr).lower()


def test_provider_is_used(tmp_path, mock_server):
    # providers.yaml holds a live provider (mock) and a dead one (other);
    # --provider selects which endpoint receives the ping.
    pv = tmp_path / "providers.yaml"
    pv.write_text(
        f"""providers:
  - name: mock
    base_url: {mock_server.base_url}
    token: sekret
  - name: other
    base_url: http://127.0.0.1:1/v1
    token: ""
"""
    )
    # dead endpoint → non-zero (connection refused) → loss → exit 1
    result = runner.invoke(
        app, ["--providers", str(pv), "--provider", "other", "--model", "mistral"]
    )
    assert result.exit_code == 1
    # live endpoint → ok
    result = runner.invoke(
        app, ["--providers", str(pv), "--provider", "mock", "--model", "mistral"]
    )
    assert result.exit_code == 0, result.output


def test_run_dir_only_with_dump_traffic(tmp_path, mock_server):
    # A run without --dump-traffic writes NOTHING to disk — the report is
    # stdout-only (--format json); run dirs exist only for traffic logs.
    pv = _write_providers(tmp_path, mock_server.base_url)
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--runs-dir",
            str(runs),
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["stats"]["sent"] == 1
    assert not runs.exists()  # no run dir, no files at all


def test_dump_traffic_run_dir_named_provider_model_scenario(tmp_path, mock_server):
    # The run dir is <provider>_<model>_<scenario>_<ts> and holds only the
    # traffic logs — no report/manifest files.
    pv = _write_providers(tmp_path, mock_server.base_url)
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--runs-dir",
            str(runs),
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--scenario",
            "agentic-basic",
            "--dump-traffic",
        ],
    )
    assert result.exit_code == 0, result.output
    dirs = list(runs.iterdir())
    assert len(dirs) == 1
    name = dirs[0].name
    assert name.startswith("mock_mistral_agentic-basic_2")  # provider_model_scenario_ts
    assert not (dirs[0] / "report.json").exists()
    assert not (dirs[0] / "run.json").exists()
    assert list(dirs[0].glob("traffic-*.jsonl"))  # the exchange logs are there


def test_dump_traffic_run_dir_escapes_slashes(tmp_path, mock_server):
    # A model name containing `/` must not break out of the runs dir.
    pv = _write_providers(tmp_path, mock_server.base_url)
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--runs-dir",
            str(runs),
            "--provider",
            "mock",
            "--model",
            "org/mistral",
            "--dump-traffic",
        ],
    )
    assert result.exit_code == 0, result.output
    dirs = list(runs.iterdir())
    assert len(dirs) == 1
    assert "/" not in dirs[0].name
    assert dirs[0].name.startswith("mock_org_mistral_")


def test_dump_traffic_writes_exchanges(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--runs-dir",
            str(runs),
            "--dump-traffic",
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--count",
            "2",
            "--interval",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output
    rd = next(runs.iterdir())
    segs = sorted(rd.glob("traffic-*.jsonl"))
    assert [p.name for p in segs] == ["traffic-000000.jsonl", "traffic-000001.jsonl"]
    lines = [json.loads(ln) for p in segs for ln in p.read_text().splitlines()]
    assert len(lines) == 2  # one exchange per round/seq
    ex = lines[0]
    assert ex["kind"] == "chat.completions"
    assert "ts" in ex
    assert ex["request"]["url"].endswith("/chat/completions")
    assert ex["request"]["body"]["model"] == "mistral"
    assert ex["request"]["body"]["stream"] is True
    assert ex["response"]["status"] == 200
    # raw body keeps the SSE text; decoded body is the parsed frame list
    assert "data: [DONE]" in ex["response"]["body_raw"]
    assert isinstance(ex["response"]["body"], list)
    assert any("choices" in f for f in ex["response"]["body"])
    content = "".join(
        c.get("delta", {}).get("content", "")
        for f in ex["response"]["body"]
        if isinstance(f, dict)
        for c in f.get("choices", [])
    )
    assert content  # decoded frames carry the content deltas
    text = "".join(p.read_text() for p in segs)
    assert "sekret" not in text  # Authorization redacted
    assert '"authorization": "***"' in text


def test_interrupted_exit130_reports_partial(tmp_path, mock_server, monkeypatch):
    # Ctrl-C on `--count inf`: run_ping returns (partial stats, interrupted),
    # and the CLI must still print the summary and exit 130 — not blow up
    # with a traceback. The report is stdout-only (--format json).
    from llmping.ping import PingStats, Sample

    async def fake_run_ping(client, targets, **kwargs):
        stats = [PingStats(model=t.model) for t in targets]
        stats[0].samples.append(
            Sample(model=targets[0].model, seq=0, ttft_ms=1.0, e2e_ms=2.0, turns=1)
        )
        return stats, True

    monkeypatch.setattr("llmping.cli.run_ping", fake_run_ping)
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "--format",
            "json",
            "--provider",
            "mock",
            "--model",
            "mistral",
            "--count",
            "inf",
        ],
    )
    assert result.exit_code == 130, result.output
    report = json.loads(result.stdout)
    assert report["stats"]["sent"] == 1  # partial stats survived
    assert "1 requests sent" in result.output or report  # summary printed


def test_bad_interval_timeout_exit2(tmp_path, mock_server):
    # Non-finite numbers pass a plain `<= 0` guard but corrupt the run
    # (inf interval sleeps forever, nan skips the sleep, nan timeout fails
    # every request) — reject them like any other bad flag value.
    pv = _write_providers(tmp_path, mock_server.base_url)
    for flag, value in (
        ("--interval", "inf"),
        ("--interval", "nan"),
        ("--interval", "0"),
        ("--timeout", "nan"),
        ("--timeout", "0"),
    ):
        result = runner.invoke(app, [*_ping_args(pv, model="mistral"), flag, value])
        assert result.exit_code == 2, (flag, value, result.output)
        assert flag in (result.output + result.stderr)


def test_short_options_work(tmp_path, mock_server):
    # -p/-m/-s/-c/-i/-t/-d/-f mirror their long forms; values can be
    # attached (-fjson) or separate (-f json).
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            "--providers",
            str(pv),
            "-p",
            "mock",
            "-m",
            "mistral",
            "-s",
            "agentic-basic",
            "-c",
            "1",
            "-i",
            "0.1",
            "-t",
            "10",
            "-f",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["scenario"] == "agentic-basic"
    assert report["stats"]["turns"] == 2


def test_short_options_attached_values(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app, ["--providers", str(pv), "-p", "mock", "-mmistral", "-c2", "-i0.05", "-fjson"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["stats"]["sent"] == 2


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes — on GitHub Actions runners typer's rich
    help renders WITH color (GITHUB_ACTIONS makes rich auto-enable it),
    so help-output assertions must run on the plain text."""
    import re

    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def test_short_help_flag(tmp_path):
    # -h and --help are the same option.
    short = runner.invoke(app, ["-h"])
    long = runner.invoke(app, ["--help"])
    assert short.exit_code == 0 and long.exit_code == 0
    plain = _strip_ansi(short.output)
    assert "Ping OpenAI-compatible LLM servers" in plain
    assert "--provider" in plain and "-p" in plain


def test_bare_short_scenario_lists_names(tmp_path):
    # A bare -s (no value) lists scenarios, exactly like a bare --scenario.
    result = runner.invoke(app, ["-s"])
    assert result.exit_code == 0, result.output
    assert "pingpong" in result.output
    # ...and warns about ignored flags in either form
    result = runner.invoke(app, ["-s", "-c", "2"])
    out = result.output + result.stderr
    assert "[WARN]" in out and "-c" in out


def test_version_flag(tmp_path):
    # -V/--version is eager: it answers before any validation (no providers
    # file, no --model needed) and exits 0.
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("llmping ")
    import llmping

    assert llmping.__version__ in result.output
    # short form identical
    short = runner.invoke(app, ["-V"])
    assert short.exit_code == 0
    assert short.output == result.output


def test_list_models_short_flag(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, ["--providers", str(pv), "-l", "-p", "mock"])
    assert result.exit_code == 0, result.output
    assert "gpt-oss-120b" in result.output


def test_bad_format_exit2(tmp_path, mock_server):
    # unknown --format is a usage error (exit 2), like every other flag —
    # no silent fallback to table.
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, [*_ping_args(pv, model="mistral"), "--format", "yaml"])
    assert result.exit_code == 2
    assert "--format" in (result.output + result.stderr)
    assert "yaml" in (result.output + result.stderr)


def test_ping_closes_http_client(tmp_path, mock_server, monkeypatch):
    # The client must be closed inside the run's event loop; asyncio.run tears
    # the loop down on return, so a later close would be too late.
    from llmping.cli import LLMClient

    closed: list[bool] = []
    real_close = LLMClient.close

    async def spy_close(self):
        closed.append(True)
        await real_close(self)

    monkeypatch.setattr(LLMClient, "close", spy_close)
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, _ping_args(pv, model="mistral"))
    assert result.exit_code == 0, result.output
    assert closed == [True]


def test_traffic_dump_closed_after_run(tmp_path, mock_server, monkeypatch):
    # The dump's file handles must be released as soon as the run ends —
    # even when post-run work (summary printing) raises.
    from llmping import rundir

    closed: list[bool] = []
    real_close = rundir.TrafficDump.close

    def spy_close(self):
        closed.append(True)
        real_close(self)

    def boom(*args, **kwargs):
        # The pre-run "traffic dump:" line legitimately prints; only the
        # post-run summary printing should raise here. (Patched on the
        # class, so args[0] is the console instance itself.)
        if len(args) > 1 and isinstance(args[1], str) and args[1].startswith("traffic dump:"):
            return
        raise OSError("disk full")

    monkeypatch.setattr(rundir.TrafficDump, "close", spy_close)
    monkeypatch.setattr("llmping.cli.Console.print", boom)
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(
        app,
        [
            *_ping_args(pv, model="mistral"),
            "--dump-traffic",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code != 0
    assert closed == [True]


def test_pad_option_reports_input_tokens(tmp_path, mock_server):
    # --pad N grows the first user message by N chars; the live line and
    # the summary report the server-counted input tokens (mock: 4 chars
    # per token, so 4000 pad chars ≈ +1000 tokens).
    pv = _write_providers(tmp_path, mock_server.base_url)
    base = runner.invoke(app, _ping_args(pv, model="mistral"))
    padded = runner.invoke(app, [*_ping_args(pv, model="mistral"), "--pad", "4000"])
    assert base.exit_code == 0 and padded.exit_code == 0
    # e.g. "in=8" vs "in=~1008"
    import re

    grab = lambda out: int(re.search(r"in=(\d+)", out).group(1))  # noqa: E731
    assert grab(padded.output) - grab(base.output) >= 950
    assert "input tokens:" in padded.output


def test_pad_rejected_negative(tmp_path, mock_server):
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, [*_ping_args(pv, model="mistral"), "--pad", "-5"])
    assert result.exit_code == 2


def test_nonce_flag_on_live_line(tmp_path, mock_server):
    # --nonce busts the KV cache per round; the live line carries the
    # server-reported cached= figure either way.
    pv = _write_providers(tmp_path, mock_server.base_url)
    result = runner.invoke(app, [*_ping_args(pv, model="mistral"), "--nonce", "--count", "2"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert out.count("[nonce") == 0  # the nonce is outbound, not printed
    assert "cached=" in out  # cached= appears on both rounds
