#!/usr/bin/env python3
"""Retrieve captions and transcripts before a query time for ablation studies."""

import json
import os
import re
from pathlib import Path
from typing import Any


def parse_query_time(query_time: str | dict) -> tuple[int, int] | None:
    """Parse query_time to (day_num, time_int_8digit).

    Handles:
      - "DAY6, 18:30:00" -> (6, 18300000)
      - {"date": "DAY6", "time": "18300000"} -> (6, 18300000)
    """
    if isinstance(query_time, dict):
        date = str(query_time.get("date") or "").strip()
        time_str = str(query_time.get("time") or "").strip()
        day_m = re.search(r"(\d+)", date)
        if not day_m or not time_str:
            return None
        day_num = int(day_m.group(1))
        time_val = int(time_str)
        if len(time_str) == 6:
            time_val *= 100
        return (day_num, time_val)

    if not isinstance(query_time, str):
        return None
    text = query_time.strip()

    m = re.match(r"^DAY(\d+)\s*,\s*(\d{1,2}):(\d{2}):(\d{2})$", text, flags=re.IGNORECASE)
    if m:
        day = int(m.group(1))
        hh, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
        return (day, hh * 1000000 + mm * 10000 + ss * 100)

    m = re.match(r"^DAY(\d+)[_ ](\d{6,8})$", text, flags=re.IGNORECASE)
    if m:
        day = int(m.group(1))
        raw = m.group(2)
        if len(raw) == 6:
            raw += "00"
        return (day, int(raw))

    return None


def _normalize_caption_day(date_val: Any) -> int:
    """Normalize caption date field to integer day number.

    30sec captions use "DAY1" strings, 3min/10min/1h use integer 1.
    """
    if isinstance(date_val, int):
        return date_val
    s = str(date_val).strip().upper()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _normalize_time(time_val: Any) -> int:
    """Normalize time to 8-digit integer."""
    t = int(time_val)
    if t < 100000000 and len(str(abs(t))) <= 6:
        t *= 100
    return t


def get_captions_before_target(
    identity: str,
    day_num: int,
    time_int: int,
    caption_root: str,
    duration: str = "3min",
    max_chars: int | None = None,
) -> str:
    """Return caption text for entries strictly before (day_num, time_int).

    Args:
        identity: e.g. "A1_JAKE"
        day_num: target day number
        time_int: 8-digit target time
        caption_root: root dir containing per-identity subdirs
        duration: one of "30sec", "3min", "10min", "1h"
        max_chars: if set, truncate from the oldest to fit
    """
    caption_file = os.path.join(caption_root, identity, f"{identity}_{duration}.json")
    if not os.path.isfile(caption_file):
        return ""

    with open(caption_file, "r") as f:
        entries = json.load(f)

    filtered = []
    for entry in entries:
        d = _normalize_caption_day(entry.get("date", 0))
        t = _normalize_time(entry.get("start_time", 0))
        if (d, t) < (day_num, time_int):
            filtered.append((d, t, entry.get("text", "").strip()))

    filtered.sort(key=lambda x: (x[0], x[1]))

    texts = [f"[DAY{d} {t:08d}] {text}" for d, t, text in filtered]

    if not texts:
        return ""

    result = "\n\n".join(texts)
    if max_chars and len(result) > max_chars:
        # Keep the most recent entries (closest to query time)
        trimmed = []
        total = 0
        for t in reversed(texts):
            if total + len(t) + 2 > max_chars and trimmed:
                break
            trimmed.append(t)
            total += len(t) + 2
        trimmed.reverse()
        result = "\n\n".join(trimmed)

    return result


def _parse_srt_timestamp_to_seconds(ts: str) -> float:
    """Parse SRT timestamp like '00:09:47,100' to seconds."""
    m = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", ts.strip())
    if not m:
        return 0.0
    h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _file_start_time_from_name(filename: str) -> tuple[int, int] | None:
    """Extract (day_num, time_int_8digit) from SRT filename.

    Example: A1_JAKE_DAY1_11000000.srt -> (1, 11000000)
    """
    m = re.search(r"DAY(\d+)[_](\d{8})\.srt$", filename, re.IGNORECASE)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _seconds_to_time_int(seconds: float) -> int:
    """Convert seconds to 8-digit time int HHMMSSHH."""
    total_cs = int(seconds * 100)
    hh = total_cs // 360000
    remainder = total_cs % 360000
    mm = remainder // 6000
    remainder = remainder % 6000
    ss = remainder // 100
    cs = remainder % 100
    return hh * 1000000 + mm * 10000 + ss * 100 + cs


def _time_int_to_seconds(time_int: int) -> float:
    """Convert 8-digit time int HHMMSSHH to seconds."""
    hh = time_int // 1000000
    remainder = time_int % 1000000
    mm = remainder // 10000
    remainder = remainder % 10000
    ss = remainder // 100
    cs = remainder % 100
    return hh * 3600 + mm * 60 + ss + cs / 100.0


def get_transcripts_before_target(
    identity: str,
    day_num: int,
    time_int: int,
    transcript_root: str,
    max_chars: int | None = None,
) -> str:
    """Return transcript text from SRT files before (day_num, time_int).

    Extracts the English line (second content line) from each bilingual subtitle block.
    """
    identity_dir = os.path.join(transcript_root, identity)
    if not os.path.isdir(identity_dir):
        return ""

    # Collect all relevant SRT files
    srt_files: list[tuple[int, int, str]] = []  # (day, file_start_time, path)
    for day_dir in sorted(os.listdir(identity_dir)):
        day_m = re.match(r"DAY(\d+)$", day_dir, re.IGNORECASE)
        if not day_m:
            continue
        d = int(day_m.group(1))
        if d > day_num:
            continue
        day_path = os.path.join(identity_dir, day_dir)
        if not os.path.isdir(day_path):
            continue
        for fname in sorted(os.listdir(day_path)):
            if not fname.endswith(".srt"):
                continue
            parsed = _file_start_time_from_name(fname)
            if not parsed:
                continue
            file_day, file_time = parsed
            # Include files from earlier days, or same-day files starting before target
            if file_day < day_num or (file_day == day_num and file_time < time_int):
                srt_files.append((file_day, file_time, os.path.join(day_path, fname)))

    srt_files.sort(key=lambda x: (x[0], x[1]))

    all_lines: list[str] = []
    for file_day, file_start, srt_path in srt_files:
        file_start_sec = _time_int_to_seconds(file_start)
        need_time_filter = (file_day == day_num)

        try:
            with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue

        # Parse SRT blocks
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            # Line 0: sequence number, Line 1: timestamps, Line 2+: content
            ts_line = lines[1]
            ts_match = re.match(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", ts_line)
            if not ts_match:
                continue

            if need_time_filter:
                start_offset_sec = _parse_srt_timestamp_to_seconds(ts_match.group(1))
                absolute_sec = file_start_sec + start_offset_sec
                absolute_time_int = _seconds_to_time_int(absolute_sec)
                if absolute_time_int >= time_int:
                    continue

            # Extract English line (the last content line with a speaker label or plain English)
            content_lines = lines[2:]
            english_line = None
            for cl in content_lines:
                # English line: contains ASCII letters and likely has a colon for speaker label
                if re.search(r"[a-zA-Z]{2,}", cl):
                    english_line = cl.strip()
            if english_line:
                all_lines.append(english_line)

    if not all_lines:
        return ""

    result = "\n".join(all_lines)
    if max_chars and len(result) > max_chars:
        # Keep the most recent lines
        trimmed = []
        total = 0
        for line in reversed(all_lines):
            if total + len(line) + 1 > max_chars and trimmed:
                break
            trimmed.append(line)
            total += len(line) + 1
        trimmed.reverse()
        result = "\n".join(trimmed)

    return result


def get_context_text(
    item: dict[str, Any],
    caption_root: str | None = None,
    caption_duration: str = "3min",
    transcript_root: str | None = None,
    max_context_chars: int = 4000,
) -> str:
    """Build context text from captions and/or transcripts for a benchmark item.

    Returns empty string if no context is available.
    """
    identity = item.get("identity") or item.get("video_id") or ""
    query_time = item.get("query_time")
    if not identity or not query_time:
        return ""

    parsed = parse_query_time(query_time)
    if not parsed:
        return ""
    day_num, time_int = parsed

    has_captions = caption_root is not None
    has_transcripts = transcript_root is not None

    if not has_captions and not has_transcripts:
        return ""

    # Allocate character budget
    if has_captions and has_transcripts:
        cap_budget = max_context_chars // 2
        trans_budget = max_context_chars - cap_budget
    elif has_captions:
        cap_budget = max_context_chars
        trans_budget = 0
    else:
        cap_budget = 0
        trans_budget = max_context_chars

    parts: list[str] = []

    if has_captions and cap_budget > 0:
        cap_text = get_captions_before_target(
            identity, day_num, time_int, caption_root, duration=caption_duration, max_chars=cap_budget
        )
        if cap_text:
            parts.append(f"Relevant context from activity captions:\n{cap_text}")

    if has_transcripts and trans_budget > 0:
        trans_text = get_transcripts_before_target(
            identity, day_num, time_int, transcript_root, max_chars=trans_budget
        )
        if trans_text:
            parts.append(f"Relevant context from dialogue transcripts:\n{trans_text}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys

    # Quick test: python context_retrieval.py <identity> <query_time>
    # e.g.: python context_retrieval.py A1_JAKE "DAY6, 18:30:00"
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <identity> <query_time> [caption_root] [transcript_root]")
        sys.exit(1)

    identity = sys.argv[1]
    qt = sys.argv[2]
    cap_root = sys.argv[3] if len(sys.argv) > 3 else None
    trans_root = sys.argv[4] if len(sys.argv) > 4 else None

    parsed = parse_query_time(qt)
    print(f"Parsed query_time: {parsed}")
    if not parsed:
        sys.exit(1)
    day, t = parsed

    if cap_root:
        caps = get_captions_before_target(identity, day, t, cap_root, max_chars=2000)
        print(f"\n=== Captions ({len(caps)} chars) ===")
        print(caps[:500] if caps else "(none)")

    if trans_root:
        trans = get_transcripts_before_target(identity, day, t, trans_root, max_chars=2000)
        print(f"\n=== Transcripts ({len(trans)} chars) ===")
        print(trans[:500] if trans else "(none)")
