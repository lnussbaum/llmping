"""Run-dir and traffic-dump tests.

A run dir exists only to hold `--dump-traffic` exchange logs — no
manifest/report files are written (the JSON report is stdout-only).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from llmping import rundir
from llmping.rundir import RunDir, TrafficDump, redact_headers


def test_run_dir_created_and_empty(tmp_path: Path):
    rd = RunDir(tmp_path, "prov_model_pingpong_20260101T120000")
    assert rd.path is not None
    assert rd.path.is_dir()
    assert list(rd.path.iterdir()) == []  # traffic only, written lazily


def test_run_dir_none_writes_nothing(tmp_path: Path):
    rd = RunDir.none()
    assert rd.path is None
    assert list(tmp_path.iterdir()) == []  # nothing created anywhere


def test_run_dir_escapes_path_separators(tmp_path: Path):
    # `/` in provider/model/scenario names must not break out of the runs
    # dir — the name is flattened to `_`.
    rd = RunDir(tmp_path, "a/b/../c_model_x_pingpong_20260101T120000")
    assert rd.path is not None
    assert rd.path.parent == tmp_path
    assert "/" not in rd.path.name
    assert rd.path.name.startswith("a_b_.._c_model_x_pingpong_")


def test_run_dir_same_second_collision_gets_suffix(tmp_path: Path, monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(rundir, "datetime", FrozenDatetime)
    first = RunDir(tmp_path, "p_m_pingpong_20260101T120000")
    second = RunDir(tmp_path, "p_m_pingpong_20260101T120000")
    third = RunDir(tmp_path, "p_m_pingpong_20260101T120000")
    assert first.path is not None and second.path is not None and third.path is not None
    assert first.path.name == "p_m_pingpong_20260101T120000"
    assert second.path.name == "p_m_pingpong_20260101T120000-1"
    assert third.path.name == "p_m_pingpong_20260101T120000-2"
    assert all(p.is_dir() for p in (first.path, second.path, third.path))


def test_traffic_dump_disabled_is_noop(tmp_path: Path):
    dump = TrafficDump.disabled()
    dump.begin("000000")
    dump.record({"kind": "request"})
    dump.close()
    assert list(tmp_path.iterdir()) == []


def test_traffic_dump_splits_per_seq(tmp_path: Path):
    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    dump = TrafficDump(tmp_path / "traffic.jsonl")
    dump.record({"kind": "no-seq"})
    dump.begin("000000")
    dump.record({"kind": "request", "path": "/chat/completions"})
    dump.begin("000001")
    dump.record({"kind": "request", "path": "/models"})
    dump.end()
    dump.record({"kind": "after-end"})
    dump.close()

    base = read(tmp_path / "traffic.jsonl")
    seq0 = read(tmp_path / "traffic-000000.jsonl")
    seq1 = read(tmp_path / "traffic-000001.jsonl")
    assert [e["kind"] for e in base] == ["no-seq", "after-end"]
    assert [e["kind"] for e in seq0] == ["request"]
    assert seq1[0]["path"] == "/models"
    assert all("ts" in e for e in base + seq0 + seq1)


def test_traffic_dump_begin_releases_previous_handle(tmp_path: Path):
    # One open handle at a time: `begin()` must close the previous unit's
    # file, or a long `--count inf --dump-traffic` run leaks one fd per
    # round until fd exhaustion turns every round into a spurious loss.
    import os

    dump = TrafficDump(tmp_path / "traffic.jsonl")
    for seq in range(50):
        dump.begin(f"{seq:06d}")
        dump.record({"kind": "request", "seq": seq})
    open_fds = len(os.listdir("/proc/self/fd"))
    assert len(dump._fh) == 1  # only the last unit still has a handle
    dump.close()
    # and the leaked-handle count matches: one fd for the dump, not fifty
    # (compare against the closed state to cancel out unrelated fds)
    closed_fds = len(os.listdir("/proc/self/fd"))
    assert open_fds - closed_fds == 1
    # every seq file still holds its record
    for seq in range(50):
        p = tmp_path / f"traffic-{seq:06d}.jsonl"
        assert p.exists() and json.loads(p.read_text())["seq"] == seq


def test_traffic_dump_creates_parent_dir(tmp_path: Path):
    dump = TrafficDump(tmp_path / "nested" / "traffic.jsonl")
    dump.record({"kind": "request"})
    dump.close()
    assert (tmp_path / "nested" / "traffic.jsonl").exists()


def test_redact_headers_masks_credentials():
    headers = {
        "Authorization": "Bearer s3cr3t",
        "x-api-key": "k",
        "Content-Type": "application/json",
        "cookie": "a=b",
    }
    masked = redact_headers(headers)
    assert masked["Authorization"] == "***"
    assert masked["x-api-key"] == "***"
    assert masked["cookie"] == "***"
    assert masked["Content-Type"] == "application/json"
