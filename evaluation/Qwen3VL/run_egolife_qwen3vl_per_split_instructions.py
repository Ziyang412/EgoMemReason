"""Qwen3-VL evaluation with per-query-type instruction prompts.

Wraps run_egolife_qwen3vl_ablation.py by injecting query_type-specific reasoning
instructions before each question.

Usage:
    python run_egolife_qwen3vl_per_split_instructions.py \
        --dataset /path/to/final_benchmark_500_apr22.json \
        --output /path/to/results.json \
        --egolife_frame_index_dir /path \
        --max_frames 256
"""

import json
import sys
import os

# Ensure relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_egolife_qwen3vl_ablation as base
from run_egolife_qwen3vl_ablation import (
    build_query_from_item,
    PROMPT_BUILDERS,
)


# ---------------------------------------------------------------------------
# Per-split instruction prompts
# ---------------------------------------------------------------------------

PER_SPLIT_INSTRUCTIONS_SHORT = {
    "Activity Pattern": (
        "Tally the relevant occurrences across days. Pick the most frequent / consistent pattern. "
        "If nothing recurs, prefer a 'no repeated' answer."
    ),
    "Cumulative State Tracking": (
        "Identify the target object visible now. Then search backward to the EARLIER moment named in the question "
        "and read off the object's location/state there. Be precise about floor (1st vs 2nd) and room."
    ),
    "Event Linking": (
        "Identify the question subtype (count, first/last, duration, cross-day) and act accordingly: "
        "'how many' = count distinct occurrences; 'first/last' = scan chronologically; "
        "'how long between X and Y' = compare day+time of each (~24h/day); 'X but not Y' = verify presence in X AND absence in Y."
    ),
    "Event Ordering": (
        "Locate each numbered event in the frames, note its day+time, sort earliest->latest, "
        "and match the sequence to one option. Distractor options often differ by one adjacent swap."
    ),
    "Spatial Preference": (
        "List all relevant occurrences with their location (room+floor+sub-area). "
        "For 'most often' pick the most frequent location; for 'where' pick the matching one. Be precise about floor."
    ),
    "Temporal Counting": (
        "Count UNIQUE physical instances across all frames; do not double-count the same item across frames. "
        "Same object in different locations = 1. Two identical-looking items in one frame = 2. "
        "If unsure between close counts, prefer the lower (overcounting is more common)."
    ),
}


PER_SPLIT_INSTRUCTIONS = {
    "Activity Pattern": (
        "This question asks about a recurring pattern across multiple days. Steps to reason:\n"
        "1. Locate every relevant occurrence across all days using the chronological frame order.\n"
        "2. Tally the count or category for each occurrence per day.\n"
        "3. Compare across days to find the most frequent / most consistent pattern.\n"
        "4. If no item recurs, the answer may indicate 'no repeated' or similar — verify by checking that each option is or isn't a recurring item.\n"
    ),
    "Cumulative State Tracking": (
        "This question asks about an object's location or state at a SPECIFIC EARLIER TIME, not at the current query time. Steps:\n"
        "1. Identify the target object (or person) referenced in the question, using its appearance in the most recent frame as a visual anchor.\n"
        "2. Search backward through the chronological frames for the referenced earlier event (e.g., 'when we had a meeting yesterday noon' -> find frames matching a meeting context on the prior day around noon).\n"
        "3. At those identified earlier frames, locate the target object and read off its position/state.\n"
        "4. Match against options. Pay attention to floor (1st vs 2nd) and room name precision.\n"
    ),
    "Event Linking": (
        "This question asks about relationships between events across time (frequency, ordering, duration). Identify the question type first:\n"
        "- 'How many times...' -> count distinct occurrences across all visible days; do not double-count overlapping clips.\n"
        "- 'When was the first/last time...' -> scan chronologically; record day + approximate time of each occurrence; pick earliest or latest.\n"
        "- 'How long was it between X and Y' -> estimate hours by comparing the day-and-time of each event (one full day ~ 24 hours).\n"
        "- 'On the day I did X, what did I do for Y?' -> first localize the day where X occurred, then find Y on that same day.\n"
        "- 'X but not Y' -> confirm the activity exists in X's window AND verify it is absent from Y's window.\n"
    ),
    "Event Ordering": (
        "This question gives 6 or 8 events; order them earliest -> latest. Steps:\n"
        "1. For each numbered event, find the frames depicting it.\n"
        "2. Note the day and approximate time of each event (use chronological frame order as your timeline).\n"
        "3. Sort the event numbers by day-and-time (earliest first).\n"
        "4. Match your sorted order to the option strings. Two options often differ only by a single adjacent swap - read carefully.\n"
    ),
    "Spatial Preference": (
        "This question asks where an action most often takes place, or which spatial location best matches a description. Steps:\n"
        "1. Identify all visible relevant occurrences of the action / object.\n"
        "2. For each, note the location (room: living room, kitchen, bedroom, courtyard, balcony; floor: 1st or 2nd; specific surface: sofa, table, etc.).\n"
        "3. Tally locations and pick the most frequent (for 'most often / preferred') or the matching one (for 'where').\n"
        "4. Be precise about floor (1st vs 2nd) and which sub-area within a room.\n"
    ),
    "Temporal Counting": (
        "This question asks how many distinct instances of an item or event you have seen UP TO NOW. Steps:\n"
        "1. Identify the target item (e.g., chairs, vases, whiteboards) - be specific about what counts (a chair vs a stool, a vase vs a cup).\n"
        "2. Scan all frames chronologically. Each unique physical instance counts once - do NOT recount the same item appearing in multiple frames.\n"
        "3. If the same object moves between locations, it still counts as ONE instance.\n"
        "4. If two visually identical items appear at the same time (e.g., two matching chairs in one frame), count them separately.\n"
        "5. If unsure between two close numbers, lean toward the lower count (overcounting is the more common error).\n"
    ),
}


# ---------------------------------------------------------------------------
# Per-split crafted ICL examples (1 worked example per query type)
# ---------------------------------------------------------------------------

PER_SPLIT_ICL = {
    "Activity Pattern": (
        "Example:\n"
        "Question: What do I most often eat for breakfast?\n"
        "Options:\n  A. Pancake\n  B. Yogurt\n  C. Toast\n  D. Congee\n  E. No repeated breakfast\n"
        "Reasoning: Day1 breakfast = pancake; Day2 = pancake; Day3 = toast; Day4 = pancake; "
        "Day5 = yogurt; Day6 = pancake; Day7 = pancake. Pancake appears 5/7 days, the most frequent.\n"
        "Answer: A\n"
    ),
    "Cumulative State Tracking": (
        "Example:\n"
        "Question: Looking at this red mug in front of me now, where was it when we had the team meeting in the main room yesterday noon?\n"
        "Options:\n  A. on the kitchen counter\n  B. in front of the whiteboard\n  C. in the bedroom\n  D. on the courtyard table\n"
        "Reasoning: Identify the red mug now. Search backward to yesterday's noon meeting frames - "
        "the mug is visible on the kitchen counter during that meeting (someone fetched it later).\n"
        "Answer: A\n"
    ),
    "Event Linking": (
        "Example:\n"
        "Question: When was the last time we made cake on this table?\n"
        "Options:\n  A. DAY3 7PM-11PM\n  B. DAY2 7PM-11PM\n  C. DAY1 11AM-3PM\n  D. DAY3 3PM-7PM\n"
        "Reasoning: Scan all cake-making events visible. I see two: DAY1 around lunch (small cake) and "
        "DAY3 in the evening (group party cake). The most recent (latest) is DAY3 7PM-11PM.\n"
        "Answer: A\n"
    ),
    "Event Ordering": (
        "Example:\n"
        "Question: Order earliest -> latest:\n  1) Eat pizza  2) Play games  3) Make cake  4) Take group photo\n"
        "Options:\n  A. 1 -> 2 -> 3 -> 4\n  B. 1 -> 3 -> 2 -> 4\n  C. 2 -> 1 -> 3 -> 4\n  D. 1 -> 2 -> 4 -> 3\n"
        "Reasoning: Locate each event in frames. Pizza = DAY1 lunch; Games = DAY2 evening; "
        "Cake = DAY3 evening; Photo = DAY4 noon. Sorted: 1, 2, 3, 4.\n"
        "Answer: A\n"
    ),
    "Spatial Preference": (
        "Example:\n"
        "Question: Where do I most often have lunch?\n"
        "Options:\n  A. in the courtyard\n  B. in the main room\n  C. in the kitchen\n  D. in the bedroom\n  E. on the second floor balcony\n"
        "Reasoning: Day1 lunch = main room; Day2 = courtyard; Day3 = courtyard; Day4 = courtyard; "
        "Day5 = main room. Courtyard appears 3/5 times, the most frequent location.\n"
        "Answer: A\n"
    ),
    "Temporal Counting": (
        "Example:\n"
        "Question: How many distinct chairs have I seen up to now?\n"
        "Options:\n  A. 4\n  B. 5\n  C. 6\n  D. 7\n  E. 8\n"
        "Reasoning: Scan all frames. Living room: 4 chairs around the table. Bedroom: 1 desk chair. "
        "Courtyard: 1 wooden chair. Same chairs reappear in many frames but each unique chair counts ONCE. "
        "Total = 4 + 1 + 1 = 6.\n"
        "Answer: C\n"
    ),
}


def get_query_type(item):
    for k in ("query_type", "task_type", "type"):
        v = item.get(k)
        if v:
            return str(v).strip()
    return None


def _build_prompt_with_instructions(instruction_dict, item, options, context_text, include_icl=False):
    """Build prompt with per-query-type instruction (long or short) prepended.
    Optionally include a per-query-type ICL example.
    """
    query = build_query_from_item(item)
    qt = get_query_type(item)
    instruction = instruction_dict.get(qt, "") if instruction_dict else ""
    icl = PER_SPLIT_ICL.get(qt, "") if include_icl else ""

    parts = []
    header = (
        "You are reviewing a week-long video log, presented as an ordered sequence of image frames. "
        "Review the frames carefully and answer the multiple-choice question.\n"
        "First state your answer as: Answer: [LETTER]\n"
        "Then provide a brief explanation of your reasoning on the next lines."
    )
    parts.append(header)

    if instruction:
        parts.append("")
        parts.append("Reasoning guide for this question type:")
        parts.append(instruction.strip())

    if icl:
        parts.append("")
        parts.append("Worked example for this question type:")
        parts.append(icl.strip())
        parts.append("")
        parts.append("Now apply the same reasoning to the actual question below.")

    if context_text:
        parts.append("")
        parts.append(context_text)

    parts.append("")
    parts.append(query)
    return "\n".join(parts)


def build_prompt_per_split(item, options, context_text):
    return _build_prompt_with_instructions(PER_SPLIT_INSTRUCTIONS, item, options, context_text)


def build_prompt_per_split_short(item, options, context_text):
    return _build_prompt_with_instructions(PER_SPLIT_INSTRUCTIONS_SHORT, item, options, context_text)


def build_prompt_per_split_icl(item, options, context_text):
    """Long instructions + crafted ICL example."""
    return _build_prompt_with_instructions(PER_SPLIT_INSTRUCTIONS, item, options, context_text, include_icl=True)


def build_prompt_per_split_short_icl(item, options, context_text):
    """Short instructions + crafted ICL example."""
    return _build_prompt_with_instructions(PER_SPLIT_INSTRUCTIONS_SHORT, item, options, context_text, include_icl=True)


def build_prompt_per_split_icl_only(item, options, context_text):
    """ICL example only, no instructions."""
    return _build_prompt_with_instructions(None, item, options, context_text, include_icl=True)


# Register prompt strategies
PROMPT_BUILDERS["per_split"] = build_prompt_per_split
PROMPT_BUILDERS["per_split_short"] = build_prompt_per_split_short
PROMPT_BUILDERS["per_split_icl"] = build_prompt_per_split_icl
PROMPT_BUILDERS["per_split_short_icl"] = build_prompt_per_split_short_icl
PROMPT_BUILDERS["per_split_icl_only"] = build_prompt_per_split_icl_only
base.PROMPT_BUILDERS = PROMPT_BUILDERS


if __name__ == "__main__":
    # Patch argparse choices to include the new strategies
    import argparse
    _orig_add_argument = argparse.ArgumentParser.add_argument
    NEW_CHOICES = (
        "per_split", "per_split_short",
        "per_split_icl", "per_split_short_icl", "per_split_icl_only",
    )

    def patched_add_argument(self, *args, **kwargs):
        if args and args[0] == "--prompt_strategy":
            choices = kwargs.get("choices")
            if choices:
                merged = list(choices)
                for c in NEW_CHOICES:
                    if c not in merged:
                        merged.append(c)
                kwargs["choices"] = merged
        return _orig_add_argument(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = patched_add_argument

    base.main()
