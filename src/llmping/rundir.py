"""Run dirs and the optional HTTP traffic dump.

A run dir (`runs/<provider>_<model>_<scenario>_<ts>/`) exists to hold the
opt-in `--dump-traffic` exchange logs — nothing else is written to disk:
the JSON report goes to stdout under `--format json`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RunDir:
    """Creates `runs/<name>/` — the per-run directory that holds the
    `--dump-traffic` exchange logs (and nothing else: the JSON report is
    stdout-only, no manifest/report files are written).

    The name is caller-built (`provider_model_scenario_<ts>`); `/` and
    other path separators in provider/model/scenario names are escaped to
    `_` so a name can't break out of the runs dir. Second-resolution
    timestamps can collide (two runs in the same second): claim the name
    atomically and fall back to `-1`, `-2`, … so concurrent runs never
    share a dir.
    """

    path: Path | None

    def __init__(self, runs_dir: Path, name: str):
        stem = _escape_name(name)
        path = runs_dir / stem
        suffix = 0
        while True:
            try:
                path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                path = runs_dir / f"{stem}-{suffix}"
        self.path = path

    @staticmethod
    def none() -> RunDir:
        """No run dir (`--dump-traffic` off): path is None, nothing is
        created — a run without a traffic dump writes no files at all."""
        rd = RunDir.__new__(RunDir)
        rd.path = None
        return rd


def _escape_name(name: str) -> str:
    """Make a run-dir name safe: path separators (and NUL) become `_`."""
    return name.replace("/", "_").replace("\\", "_").replace("\0", "_")


_SENSITIVE_HEADERS = {
    "authorization",
    "x-api-key",
    "api-key",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "proxy-authenticate",
    "www-authenticate",
}


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Mask credential-bearing header values for the HTTP dump."""
    return {k: ("***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()}


class TrafficDump:
    """Generic exchange capture: one self-describing JSONL line per request.

    Records go to the given path (default `<run dir>/traffic.jsonl` for
    records outside a seq). For `ping`, the CLI calls `begin("<seq>")` once
    per round so each seq gets its own file `traffic-<seq>.jsonl`.
    Credential-bearing headers are masked; with no path it's a no-op
    (`TrafficDump.disabled()` — a run without --dump-traffic writes
    nothing at all).
    """

    def __init__(self, path: Path | None):
        self._base = path
        self._unit: str | None = None
        self._fh: dict[Path, TextIO] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def disabled() -> TrafficDump:
        return TrafficDump(None)

    def begin(self, unit: str) -> None:
        """Route subsequent records to `traffic-<unit>.jsonl` beside base.

        Closes the previous unit's handle first: rounds are sequential, so
        at most one handle stays open — a `--count inf --dump-traffic` run
        must not leak one fd per round (fd exhaustion then turns every
        round into a spurious loss)."""
        self._close_open()
        self._unit = unit

    def end(self) -> None:
        self._unit = None

    def _close_open(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()

    def _path(self) -> Path | None:
        if self._base is None:
            return None
        if self._unit is None:
            return self._base
        return self._base.with_name(f"{self._base.stem}-{self._unit}{self._base.suffix}")

    def record(self, exchange: dict[str, Any]) -> None:
        path = self._path()
        if path is None:
            return
        fh = self._fh.get(path)
        if fh is None:
            fh = path.open("a", encoding="utf-8")
            self._fh[path] = fh
        fh.write(json.dumps({"ts": now_iso(), **exchange}, ensure_ascii=False) + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh = {}
