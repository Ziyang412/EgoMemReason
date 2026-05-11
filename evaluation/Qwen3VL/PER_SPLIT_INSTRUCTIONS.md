# Per-Split Reasoning Instructions

Per-query-type instruction prompts injected into the model's input to guide
reasoning. Used by the `per_split` prompt strategy in
`run_egolife_qwen3vl_per_split_instructions.py`.

Each instruction is prepended to the standard query as a "Reasoning guide for
this question type" block, between the system header and the actual question.

## Results summary (Qwen3-VL-8B, 256 frames, apr22 benchmark)

| Query Type                | Baseline | Per-split | Δ      |
| ------------------------- | -------- | --------- | ------ |
| Activity Pattern          | 42.00%   | 44.00%    | +2.00  |
| Cumulative State Tracking | 35.00%   | 31.00%    | -4.00  |
| Event Linking             | 21.00%   | 20.00%    | -1.00  |
| Event Ordering            | 39.00%   | 39.00%    |  0.00  |
| Spatial Preference        | 40.00%   | 38.00%    | -2.00  |
| Temporal Counting         | 28.00%   | 22.00%    | -6.00  |
| **Overall**               | **32.80%** | **30.60%** | **-2.20** |

The verbose instructions hurt the 8B overall — likely because long procedural
prompts consume context and confuse a smaller model. May benefit larger models
(32B / MoE-30B). Worth trying shorter (1-2 line) variants or combining with ICL.

---

## 1. Activity Pattern

```
This question asks about a recurring pattern across multiple days. Steps to reason:
1. Locate every relevant occurrence across all days using the chronological frame order.
2. Tally the count or category for each occurrence per day.
3. Compare across days to find the most frequent / most consistent pattern.
4. If no item recurs, the answer may indicate 'no repeated' or similar — verify by checking that each option is or isn't a recurring item.
```

## 2. Cumulative State Tracking

```
This question asks about an object's location or state at a SPECIFIC EARLIER TIME, not at the current query time. Steps:
1. Identify the target object (or person) referenced in the question, using its appearance in the most recent frame as a visual anchor.
2. Search backward through the chronological frames for the referenced earlier event (e.g., 'when we had a meeting yesterday noon' -> find frames matching a meeting context on the prior day around noon).
3. At those identified earlier frames, locate the target object and read off its position/state.
4. Match against options. Pay attention to floor (1st vs 2nd) and room name precision.
```

## 3. Event Linking

```
This question asks about relationships between events across time (frequency, ordering, duration). Identify the question type first:
- 'How many times...' -> count distinct occurrences across all visible days; do not double-count overlapping clips.
- 'When was the first/last time...' -> scan chronologically; record day + approximate time of each occurrence; pick earliest or latest.
- 'How long was it between X and Y' -> estimate hours by comparing the day-and-time of each event (one full day ~ 24 hours).
- 'On the day I did X, what did I do for Y?' -> first localize the day where X occurred, then find Y on that same day.
- 'X but not Y' -> confirm the activity exists in X's window AND verify it is absent from Y's window.
```

## 4. Event Ordering

```
This question gives 6 or 8 events; order them earliest -> latest. Steps:
1. For each numbered event, find the frames depicting it.
2. Note the day and approximate time of each event (use chronological frame order as your timeline).
3. Sort the event numbers by day-and-time (earliest first).
4. Match your sorted order to the option strings. Two options often differ only by a single adjacent swap - read carefully.
```

## 5. Spatial Preference

```
This question asks where an action most often takes place, or which spatial location best matches a description. Steps:
1. Identify all visible relevant occurrences of the action / object.
2. For each, note the location (room: living room, kitchen, bedroom, courtyard, balcony; floor: 1st or 2nd; specific surface: sofa, table, etc.).
3. Tally locations and pick the most frequent (for 'most often / preferred') or the matching one (for 'where').
4. Be precise about floor (1st vs 2nd) and which sub-area within a room.
```

## 6. Temporal Counting

```
This question asks how many distinct instances of an item or event you have seen UP TO NOW. Steps:
1. Identify the target item (e.g., chairs, vases, whiteboards) - be specific about what counts (a chair vs a stool, a vase vs a cup).
2. Scan all frames chronologically. Each unique physical instance counts once - do NOT recount the same item appearing in multiple frames.
3. If the same object moves between locations, it still counts as ONE instance.
4. If two visually identical items appear at the same time (e.g., two matching chairs in one frame), count them separately.
5. If unsure between two close numbers, lean toward the lower count (overcounting is the more common error).
```

---

## Per-Split Crafted ICL Examples

Used by `per_split_icl`, `per_split_short_icl`, and `per_split_icl_only`
strategies. Each example is a short worked Q+reasoning+answer demo for that
query type. Stored in `PER_SPLIT_ICL` dict.

### Activity Pattern
```
Q: What do I most often eat for breakfast?
Options: A. Pancake / B. Yogurt / C. Toast / D. Congee / E. No repeated breakfast
Reasoning: Day1=pancake, Day2=pancake, Day3=toast, Day4=pancake, Day5=yogurt,
Day6=pancake, Day7=pancake. Pancake 5/7 (most frequent).
Answer: A
```

### Cumulative State Tracking
```
Q: Looking at this red mug now, where was it during yesterday's noon meeting?
Options: A. on the kitchen counter / B. in front of the whiteboard / C. in the bedroom / D. on the courtyard table
Reasoning: Identify the mug now. Search backward to yesterday noon meeting frames -
mug was on the kitchen counter then.
Answer: A
```

### Event Linking
```
Q: When was the last time we made cake on this table?
Options: A. DAY3 7PM-11PM / B. DAY2 7PM-11PM / C. DAY1 11AM-3PM / D. DAY3 3PM-7PM
Reasoning: Two cake events: DAY1 lunch (small cake), DAY3 evening (party). Latest = DAY3 7PM-11PM.
Answer: A
```

### Event Ordering
```
Q: Order earliest -> latest: 1) Eat pizza  2) Play games  3) Make cake  4) Take group photo
Options: A. 1->2->3->4 / B. 1->3->2->4 / C. 2->1->3->4 / D. 1->2->4->3
Reasoning: Pizza=Day1 lunch, Games=Day2 evening, Cake=Day3 evening, Photo=Day4 noon.
Answer: A
```

### Spatial Preference
```
Q: Where do I most often have lunch?
Options: A. courtyard / B. main room / C. kitchen / D. bedroom / E. balcony
Reasoning: Day1=main room, Day2=courtyard, Day3=courtyard, Day4=courtyard, Day5=main room.
Courtyard 3/5 (most frequent).
Answer: A
```

### Temporal Counting
```
Q: How many distinct chairs have I seen up to now?
Options: A. 4 / B. 5 / C. 6 / D. 7 / E. 8
Reasoning: Living room: 4, bedroom desk chair: 1, courtyard chair: 1.
Same chairs reappear in many frames but each unique chair counts ONCE.
Total = 6.
Answer: C
```

---

## Short variants (1-2 lines per type)

Used by the `per_split_short` prompt strategy.

| Type | Short instruction |
| --- | --- |
| Activity Pattern | Tally relevant occurrences across days. Pick the most frequent / consistent pattern. If nothing recurs, prefer 'no repeated'. |
| Cumulative State Tracking | Identify the target object visible now. Then search backward to the EARLIER moment named in the question and read off its location/state there. Be precise about floor (1st vs 2nd) and room. |
| Event Linking | Identify the subtype: 'how many'=count distinct occurrences; 'first/last'=scan chronologically; 'how long between X and Y'=compare day+time (~24h/day); 'X but not Y'=verify presence in X AND absence in Y. |
| Event Ordering | Locate each numbered event, note day+time, sort earliest->latest, match the sequence to one option. Distractors often differ by one adjacent swap. |
| Spatial Preference | List relevant occurrences with location (room+floor+sub-area). 'Most often'=most frequent; 'where'=match. Be precise about floor. |
| Temporal Counting | Count UNIQUE physical instances; do not double-count same item across frames. Same object in different locations=1. Two identical-looking items in one frame=2. If unsure between close counts, prefer the lower. |

---

## Usage

```bash
# Long instructions (per_split)
python run_egolife_qwen3vl_per_split_instructions.py \
    --dataset /path/to/final_benchmark_500_apr22.json \
    --output /path/to/results.json \
    --egolife_frame_index_dir /path \
    --max_frames 256 \
    --prompt_strategy per_split

# Short instructions (per_split_short)
python run_egolife_qwen3vl_per_split_instructions.py \
    --dataset /path/to/final_benchmark_500_apr22.json \
    --output /path/to/results.json \
    --egolife_frame_index_dir /path \
    --max_frames 256 \
    --prompt_strategy per_split_short

# Long instructions + crafted ICL example
python ... --prompt_strategy per_split_icl

# Short instructions + crafted ICL example
python ... --prompt_strategy per_split_short_icl

# ICL example only (no instructions)
python ... --prompt_strategy per_split_icl_only

# Wrapper scripts for GPUs 0-3 (chunked, then holds GPUs)
bash run_apr22_qwen8b_per_split_instructions_gpu0-3_then_hold.sh   # long
bash run_apr22_qwen8b_per_split_short_gpu0-3_then_hold.sh          # short
bash run_apr22_qwen8b_per_split_icl_both_gpu0-3_then_hold.sh       # short_icl + icl, sequential
```

To edit the instructions, modify the `PER_SPLIT_INSTRUCTIONS` (long),
`PER_SPLIT_INSTRUCTIONS_SHORT` (short), or `PER_SPLIT_ICL` (worked examples)
dicts in `run_egolife_qwen3vl_per_split_instructions.py`.
