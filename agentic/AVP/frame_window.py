"""Frame selection helpers for the agentic Gemini pipeline.

Reuses primitives from evaluation/Gemini/eval_gemini_frames.py instead of
re-implementing parsing or uniform sampling.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GEMINI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "Gemini"))
if _GEMINI_DIR not in sys.path:
    sys.path.insert(0, _GEMINI_DIR)

from eval_gemini_frames import (  # noqa: E402
    parse_day_num,
    parse_query_time,
    select_frames_before_query,
    uniform_sample,
)


def parse_region_endpoint(endpoint: str) -> Tuple[int, int]:
    """Parse 'DAY3, 09:00:00' -> (3, 9000000) using the existing parser."""
    return parse_query_time(endpoint)


def _entry_key(entry: Dict) -> Optional[Tuple[int, int]]:
    day_num = parse_day_num(entry.get("day", ""))
    if day_num is None:
        return None
    t = entry.get("time")
    if t is None:
        return None
    try:
        return day_num, int(t)
    except (TypeError, ValueError):
        return None


def count_frames_before_query(
    frames_by_identity: Dict[str, List[Dict]],
    identity: str,
    query_day_num: int,
    query_time_int: int,
) -> int:
    entries = frames_by_identity.get(identity, [])
    n = 0
    for ent in entries:
        key = _entry_key(ent)
        if key is None:
            continue
        d, t = key
        if d > query_day_num:
            continue
        if d == query_day_num and t > query_time_int:
            continue
        n += 1
    return n


def days_spanned_before_query(
    frames_by_identity: Dict[str, List[Dict]],
    identity: str,
    query_day_num: int,
    query_time_int: int,
) -> str:
    entries = frames_by_identity.get(identity, [])
    days = set()
    for ent in entries:
        key = _entry_key(ent)
        if key is None:
            continue
        d, t = key
        if d > query_day_num:
            continue
        if d == query_day_num and t > query_time_int:
            continue
        days.add(d)
    if not days:
        return "no days available"
    lo, hi = min(days), max(days)
    if lo == hi:
        return f"DAY{lo}"
    return f"DAY{lo}..DAY{hi}"


def format_frame_timestamp(day_num: int, time_int: int) -> str:
    s = f"{int(time_int):08d}"
    h = int(s[0:2])
    mi = int(s[2:4])
    sec = int(s[4:6])
    return f"DAY{day_num}, {h:02d}:{mi:02d}:{sec:02d}"


def select_frames_uniform_before_query(
    frames_by_identity: Dict[str, List[Dict]],
    identity: str,
    query_day_num: int,
    query_time_int: int,
    max_frames: int,
) -> List[Tuple[int, int, str]]:
    """Uniform sampling across all frames up to query time.

    Mirrors select_frames_before_query but returns (day, time, path) tuples
    so the caller can produce timestamp strings for the observer prompt.
    """
    entries = frames_by_identity.get(identity, [])
    eligible: List[Tuple[int, int, str]] = []
    for ent in entries:
        key = _entry_key(ent)
        if key is None:
            continue
        d, t = key
        if d > query_day_num:
            continue
        if d == query_day_num and t > query_time_int:
            continue
        p = ent.get("path")
        if isinstance(p, str) and p:
            eligible.append((d, t, p))
    if not eligible:
        return []
    eligible.sort(key=lambda x: (x[0], x[1]))
    if max_frames is None or max_frames <= 0:
        sampled = eligible
    else:
        # uniform_sample over the path list, then re-attach timestamps.
        path_list = [p for _, _, p in eligible]
        sampled_paths = uniform_sample(path_list, max_frames)
        # Re-zip — uniform_sample is order-preserving and indexes into eligible.
        path_to_meta = {p: (d, t) for d, t, p in eligible}
        sampled = [(path_to_meta[p][0], path_to_meta[p][1], p) for p in sampled_paths]
    return [(d, t, p) for d, t, p in sampled if os.path.exists(p)]


def select_frames_in_regions(
    frames_by_identity: Dict[str, List[Dict]],
    identity: str,
    regions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    query_day_num: int,
    query_time_int: int,
    max_frames: int,
) -> List[Tuple[int, int, str]]:
    """Select frames whose (day, time) falls inside any [start, end] region.

    All endpoints are (day_num, time_int) pairs. Intervals are inclusive.
    Anything past the query time is dropped to preserve causality.
    """
    entries = frames_by_identity.get(identity, [])
    if not entries or not regions:
        return []

    def _within(day: int, time_int: int) -> bool:
        for (sd, st), (ed, et) in regions:
            if (day, time_int) < (sd, st):
                continue
            if (day, time_int) > (ed, et):
                continue
            return True
        return False

    eligible: List[Tuple[int, int, str]] = []
    for ent in entries:
        key = _entry_key(ent)
        if key is None:
            continue
        d, t = key
        if d > query_day_num or (d == query_day_num and t > query_time_int):
            continue
        if not _within(d, t):
            continue
        p = ent.get("path")
        if isinstance(p, str) and p:
            eligible.append((d, t, p))

    if not eligible:
        return []
    eligible.sort(key=lambda x: (x[0], x[1]))
    if max_frames is None or max_frames <= 0:
        sampled = eligible
    else:
        path_list = [p for _, _, p in eligible]
        sampled_paths = uniform_sample(path_list, max_frames)
        path_to_meta = {p: (d, t) for d, t, p in eligible}
        sampled = [(path_to_meta[p][0], path_to_meta[p][1], p) for p in sampled_paths]
    return [(d, t, p) for d, t, p in sampled if os.path.exists(p)]


# Re-export so callers can stay inside this module.
__all__ = [
    "parse_query_time",
    "parse_region_endpoint",
    "count_frames_before_query",
    "days_spanned_before_query",
    "format_frame_timestamp",
    "select_frames_uniform_before_query",
    "select_frames_in_regions",
    "select_frames_before_query",
]
