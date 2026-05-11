"""Prompts for the agentic Gemini pipeline.

Adapted from Salesforce ActiveVideoPerception (avp/prompt.py) but rewritten
for the EgoLife frame-index setting where timestamps are `DAYn, HH:MM:SS`
strings rather than seconds-from-start.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str) -> Optional[Dict]:
    if not text:
        return None
    raw = text.strip()
    m = _JSON_FENCE_RE.search(raw)
    candidate = m.group(1) if m else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _format_options(options: object) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    if isinstance(options, list):
        lines = []
        for i, opt in enumerate(options):
            if isinstance(opt, dict) and opt.get("id") and opt.get("text"):
                lines.append(f"{opt['id']}. {opt['text']}")
            elif isinstance(opt, str):
                lines.append(opt)
        return "\n".join(lines)
    return ""


def _format_evidence(evidence: List[Dict]) -> str:
    if not evidence:
        return "(none yet — this is the first observation round)"
    lines = []
    for round_idx, round_evidence in enumerate(evidence, start=1):
        lines.append(f"Round {round_idx} observations:")
        items = round_evidence.get("key_evidence") or []
        if not items:
            lines.append("  (no concrete evidence extracted)")
        for item in items:
            ts = item.get("timestamp", "?")
            desc = item.get("description", "")
            lines.append(f"  - [{ts}] {desc}")
        summary = round_evidence.get("summary")
        if summary:
            lines.append(f"  summary: {summary}")
    return "\n".join(lines)


PLANNER_SYSTEM = """You are the planning module of an agentic video question
answering system operating on a first-person (egocentric) life-logging dataset
called EgoLife. The frames you can observe are pre-extracted and indexed in
chronological order per identity. Each frame has an absolute timestamp of the
form DAYn, HH:MM:SS (for example DAY3, 14:22:00). The recording starts on
DAY1 around 11:00:00.

Your job is to choose ONE observation step per round. The observation is then
executed by another module that selects frames matching your plan and shows
them to a vision model, which extracts evidence. After observation, a
reflector decides whether enough evidence has been collected. If not, you will
be called again with the evidence so far.

You have two observation modes:
  - "uniform": uniformly sample frames across the entire history available
    before the query time. Best when the question is about long-range
    aggregate patterns or when you have no idea where to look yet.
  - "region": restrict sampling to one or more time windows. Each region is a
    [start, end] pair of DAYn, HH:MM:SS strings. Best when the question
    references specific events or when prior evidence already pointed at a
    rough time. End must be strictly later than start. Regions must be inside
    the available history (i.e., end <= query_time).

You also choose:
  - "max_frames": integer 4..1024. This is the per-observation budget. There
    is also a TOTAL budget across all rounds (given to you below as
    "remaining_frame_budget"); your max_frames will be silently clamped to
    the remaining budget. Very sparse sampling is fine and often better:
    8 frames spread across a whole week can reveal an aggregate pattern, and
    16-32 frames over a single hour is plenty for a localized event. Save
    budget for later rounds when round 1 is exploratory.
  - "frame_size": integer side length in pixels (256, 384, or 512). Higher =
    more spatial detail per frame, fewer frames you should request.
  - "focus": short string telling the observer what to look for in this
    observation. Be specific.

Return strictly a JSON object matching this schema, no prose, no code fence:
{
  "reasoning": "one or two sentences explaining your choice",
  "step": {
    "load_mode": "uniform" | "region",
    "regions": [["DAYn, HH:MM:SS", "DAYn, HH:MM:SS"], ...],
    "max_frames": <int>,
    "frame_size": <int>,
    "focus": "<short string>"
  },
  "completion_criteria": "what evidence would let the reflector stop the loop"
}

Rules:
- Exactly one step per call.
- If load_mode is "uniform", "regions" must be [].
- If load_mode is "region", "regions" must have at least one window.
- Do NOT pick an answer letter. You only plan observations.
- Use evidence from prior rounds to refine: if you already know roughly when
  something happened, switch to region mode. If prior rounds returned nothing
  useful, broaden the search.
"""


def build_plan_prompt(
    *,
    question: str,
    options: object,
    query_type: str,
    query_time: str,
    total_frames_available: int,
    days_spanned: str,
    round_idx: int,
    max_rounds: int,
    total_frame_budget: int,
    remaining_frame_budget: int,
    evidence: List[Dict],
    missing_hint: Optional[str],
) -> str:
    options_block = _format_options(options)
    evidence_block = _format_evidence(evidence)
    missing_block = (
        f"Reflector said the missing information is: {missing_hint}"
        if missing_hint
        else "(no reflector hint yet)"
    )
    return (
        PLANNER_SYSTEM
        + "\n\n--- Current task ---\n"
        + f"Question: {question}\n"
        + f"Options:\n{options_block}\n"
        + f"Query type: {query_type}\n"
        + f"Query time (do not look past this): {query_time}\n"
        + f"History available: ~{total_frames_available} frames spanning {days_spanned}\n"
        + f"Round {round_idx + 1} of {max_rounds}\n"
        + f"Total frame budget across all rounds: {total_frame_budget}\n"
        + f"Remaining frame budget for this and later rounds: {remaining_frame_budget}\n"
        + f"\nEvidence so far:\n{evidence_block}\n"
        + f"\n{missing_block}\n"
        + "\nReturn the JSON plan now."
    )


OBSERVER_SYSTEM = """You are the observation module of an agentic video QA
system on the EgoLife egocentric dataset. You receive a small set of
chronologically ordered frames sampled according to a plan, plus the question
and what to focus on. Each frame's absolute timestamp follows DAYn, HH:MM:SS
and the frames you see are listed below in temporal order.

Extract concrete, query-relevant evidence. Do NOT pick an answer letter — that
is a separate module's job.

Return strictly a JSON object matching this schema, no prose, no code fence:
{
  "key_evidence": [
    {"timestamp": "DAYn, HH:MM:SS", "description": "<what happened>"},
    ...
  ],
  "reasoning": "how the evidence connects to the question",
  "summary": "one or two sentence compact summary of what these frames show"
}

Rules:
- Only include evidence you can actually verify in the shown frames.
- If you saw nothing relevant, return key_evidence: [] and say so in summary.
- Timestamps must be in DAYn, HH:MM:SS form.
"""


def build_observe_prompt(
    *,
    question: str,
    query_time: str,
    focus: str,
    frame_timestamps: List[str],
) -> str:
    ts_block = "\n".join(f"  frame {i + 1}: {t}" for i, t in enumerate(frame_timestamps))
    return (
        OBSERVER_SYSTEM
        + "\n\n--- Current observation ---\n"
        + f"Question: {question}\n"
        + f"Query time (do not assume any frame is past this): {query_time}\n"
        + f"Focus from the planner: {focus}\n"
        + f"\nThe following {len(frame_timestamps)} frames are provided in order:\n{ts_block}\n"
        + "\nReturn the JSON evidence now."
    )


REFLECTOR_SYSTEM = """You are the reflection module of an agentic video QA
system. You see the question, the options, and the evidence collected over
one or more observation rounds. Decide whether the evidence is sufficient to
answer with high confidence.

Return strictly a JSON object matching this schema, no prose, no code fence:
{
  "sufficient": true | false,
  "confidence": <float between 0 and 1>,
  "missing": "<what is still missing if not sufficient, else empty string>",
  "rationale": "<one or two sentences>"
}

Rules:
- "sufficient": true means the synthesizer can pick the right MCQ option
  with high confidence. Be conservative — false is fine when in doubt.
- If you set sufficient: false, the "missing" field MUST describe a concrete
  next thing to look for (e.g. "a specific timestamp for when X happened",
  "whether activity Y also occurred on day 5").
"""


def build_reflect_prompt(
    *,
    question: str,
    options: object,
    evidence: List[Dict],
    rounds_used: int,
    max_rounds: int,
) -> str:
    options_block = _format_options(options)
    evidence_block = _format_evidence(evidence)
    return (
        REFLECTOR_SYSTEM
        + "\n\n--- Current task ---\n"
        + f"Question: {question}\n"
        + f"Options:\n{options_block}\n"
        + f"Rounds used so far: {rounds_used} of {max_rounds}\n"
        + f"\nEvidence:\n{evidence_block}\n"
        + "\nReturn the JSON reflection now."
    )


SYNTHESIZER_SYSTEM = """You are the synthesizer of an agentic video QA system.
You see the question, the options, and the evidence collected over one or
more observation rounds. Pick the single best option letter.

Return strictly a JSON object matching this schema, no prose, no code fence:
{
  "selected_option": "A" | "B" | "C" | "D" | "E" | "F",
  "confidence": <float between 0 and 1>,
  "rationale": "<one or two sentences citing evidence>"
}

Rules:
- Pick exactly one letter from the options actually listed.
- Base your answer on the evidence; if evidence is thin, pick the most
  plausible option anyway and lower the confidence.
- Output only the JSON object.
"""


def build_synthesize_prompt(
    *,
    question: str,
    options: object,
    evidence: List[Dict],
) -> str:
    options_block = _format_options(options)
    evidence_block = _format_evidence(evidence)
    return (
        SYNTHESIZER_SYSTEM
        + "\n\n--- Current task ---\n"
        + f"Question: {question}\n"
        + f"Options:\n{options_block}\n"
        + f"\nEvidence:\n{evidence_block}\n"
        + "\nReturn the JSON answer now."
    )
