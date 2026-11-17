"""Metrics helpers. Ping reporting uses the ping(8)-style
min/avg/max/mdev summary — deliberately the whole module."""

from __future__ import annotations

from typing import TypedDict


class LatencyStats(TypedDict):
    """min/avg/max/mdev in ms; every field is None when there are no samples."""

    min: float | None
    avg: float | None
    max: float | None
    mdev: float | None


def ping_stats(values: list[float]) -> LatencyStats:
    """Ping(8)-style min/avg/max/mdev for the summary block."""
    if not values:
        return {"min": None, "avg": None, "max": None, "mdev": None}
    avg = sum(values) / len(values)
    mdev = sum(abs(v - avg) for v in values) / len(values)
    return {"min": min(values), "avg": avg, "max": max(values), "mdev": mdev}
