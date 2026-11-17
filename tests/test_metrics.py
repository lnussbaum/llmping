"""Metrics unit tests."""

from __future__ import annotations

from llmping.metrics import ping_stats


def test_ping_stats():
    s = ping_stats([10.0, 20.0])
    assert s["min"] == 10.0
    assert s["max"] == 20.0
    assert s["avg"] == 15.0
    assert s["mdev"] == 5.0


def test_ping_stats_empty():
    s = ping_stats([])
    assert s["min"] is None
    assert s["avg"] is None
    assert s["max"] is None
    assert s["mdev"] is None
