"""Plan -> observe -> reflect agent loop, frame-index edition.

Inspired by Salesforce ActiveVideoPerception (avp/main.py Controller.run).
"""
from __future__ import annotations

import io
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GEMINI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "Gemini"))
if _GEMINI_DIR not in sys.path:
    sys.path.insert(0, _GEMINI_DIR)

from eval_gemini_frames import mime_for_path  # noqa: E402

from frame_window import (  # noqa: E402
    count_frames_before_query,
    days_spanned_before_query,
    format_frame_timestamp,
    parse_query_time,
    parse_region_endpoint,
    select_frames_in_regions,
    select_frames_uniform_before_query,
)
from prompts import (  # noqa: E402
    build_observe_prompt,
    build_plan_prompt,
    build_reflect_prompt,
    build_synthesize_prompt,
    parse_json_response,
)


VALID_FRAME_SIZES = (256, 384, 512)
DEFAULT_FRAME_SIZE = 384
MIN_FRAMES_PER_OBSERVATION = 4
MAX_FRAMES_PER_OBSERVATION = 1024
DEFAULT_TOTAL_FRAME_BUDGET = 1024


def _frame_bytes_for_model(path: str, frame_size: int) -> Tuple[bytes, str]:
    if frame_size is None or frame_size <= 0:
        with open(path, "rb") as f:
            return f.read(), mime_for_path(path)
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required for frame resizing. Install with `pip install Pillow`."
        ) from e
    with Image.open(path) as img:
        img = img.convert("RGB")
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        img = img.resize((frame_size, frame_size), resample)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


def call_gemini_text(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> str:
    config = types.GenerateContentConfig(temperature=temperature)
    response = client.models.generate_content(
        model=model,
        contents=[types.Part(text=prompt)],
        config=config,
    )
    return (response.text or "").strip()


def call_gemini_multimodal(
    client: genai.Client,
    model: str,
    prompt: str,
    frame_paths: List[str],
    frame_size: int,
) -> str:
    parts: List[types.Part] = [types.Part(text=prompt)]
    for path in frame_paths:
        data, mime_type = _frame_bytes_for_model(path, frame_size)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
    config = types.GenerateContentConfig(
        temperature=0.0,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    )
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )
    return (response.text or "").strip()


def _coerce_frame_size(value: object, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v in VALID_FRAME_SIZES:
        return v
    # Snap to nearest valid size.
    return min(VALID_FRAME_SIZES, key=lambda x: abs(x - v))


def _coerce_max_frames(value: object, fallback: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(MIN_FRAMES_PER_OBSERVATION, min(MAX_FRAMES_PER_OBSERVATION, v))


def _normalize_plan(
    plan: Optional[Dict],
    fallback_max_frames: int,
    fallback_frame_size: int,
) -> Dict:
    """Normalize a planner response into a known-good shape.

    Falls back to uniform sampling when the planner output is unusable.
    """
    if not isinstance(plan, dict):
        return {
            "load_mode": "uniform",
            "regions": [],
            "max_frames": fallback_max_frames,
            "frame_size": fallback_frame_size,
            "focus": "general overview",
            "_fallback": "plan_not_dict",
        }

    step = plan.get("step") if isinstance(plan.get("step"), dict) else plan
    load_mode = str(step.get("load_mode", "uniform")).strip().lower()
    if load_mode not in ("uniform", "region"):
        load_mode = "uniform"

    raw_regions = step.get("regions") or []
    parsed_regions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if isinstance(raw_regions, list):
        for r in raw_regions:
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                continue
            try:
                s_day, s_time = parse_region_endpoint(str(r[0]))
                e_day, e_time = parse_region_endpoint(str(r[1]))
            except Exception:
                continue
            if (e_day, e_time) <= (s_day, s_time):
                continue
            parsed_regions.append(((s_day, s_time), (e_day, e_time)))

    if load_mode == "region" and not parsed_regions:
        # Planner asked for region mode but gave us nothing parseable. Fall
        # back to uniform so we don't waste the round.
        load_mode = "uniform"

    return {
        "load_mode": load_mode,
        "regions": parsed_regions,
        "raw_regions": raw_regions if isinstance(raw_regions, list) else [],
        "max_frames": _coerce_max_frames(step.get("max_frames"), fallback_max_frames),
        "frame_size": _coerce_frame_size(step.get("frame_size"), fallback_frame_size),
        "focus": str(step.get("focus") or "general overview"),
        "reasoning": str(plan.get("reasoning") or ""),
        "completion_criteria": str(plan.get("completion_criteria") or ""),
    }


def _validate_options_letters(options: object) -> List[str]:
    if isinstance(options, dict):
        return [str(k).strip().upper() for k in options.keys() if str(k).strip()]
    return ["A", "B", "C", "D", "E", "F"]


class AgenticEvaluator:
    def __init__(
        self,
        *,
        client: genai.Client,
        planner_model: str,
        observer_model: str,
        reflector_model: str,
        synthesizer_model: str,
        max_rounds: int,
        default_max_frames: int,
        default_frame_size: int,
        total_frame_budget: int,
    ) -> None:
        self.client = client
        self.planner_model = planner_model
        self.observer_model = observer_model
        self.reflector_model = reflector_model
        self.synthesizer_model = synthesizer_model
        self.max_rounds = max(1, int(max_rounds))
        self.default_max_frames = default_max_frames
        self.default_frame_size = default_frame_size
        self.total_frame_budget = max(MIN_FRAMES_PER_OBSERVATION, int(total_frame_budget))

    def _plan(self, ctx: Dict) -> Tuple[Dict, str, float]:
        remaining = max(MIN_FRAMES_PER_OBSERVATION, ctx["remaining_budget"])
        prompt = build_plan_prompt(
            question=ctx["question"],
            options=ctx["options"],
            query_type=ctx["query_type"],
            query_time=ctx["query_time"],
            total_frames_available=ctx["total_frames"],
            days_spanned=ctx["days_spanned"],
            round_idx=ctx["round_idx"],
            max_rounds=self.max_rounds,
            total_frame_budget=self.total_frame_budget,
            remaining_frame_budget=remaining,
            evidence=ctx["evidence"],
            missing_hint=ctx.get("missing_hint"),
        )
        t0 = time.time()
        raw = call_gemini_text(self.client, self.planner_model, prompt)
        latency = round(time.time() - t0, 3)
        parsed = parse_json_response(raw)
        plan = _normalize_plan(parsed, self.default_max_frames, self.default_frame_size)
        # Hard cap: planner's max_frames cannot exceed the remaining total budget.
        if plan["max_frames"] > remaining:
            plan["max_frames_requested"] = plan["max_frames"]
            plan["max_frames"] = remaining
            plan["budget_clamped"] = True
        return plan, raw, latency

    def _observe(
        self,
        plan: Dict,
        ctx: Dict,
        frames_by_identity: Dict,
    ) -> Tuple[Dict, str, List[str], float]:
        identity = ctx["identity"]
        q_day = ctx["query_day_num"]
        q_time = ctx["query_time_int"]
        if plan["load_mode"] == "region":
            selected = select_frames_in_regions(
                frames_by_identity=frames_by_identity,
                identity=identity,
                regions=plan["regions"],
                query_day_num=q_day,
                query_time_int=q_time,
                max_frames=plan["max_frames"],
            )
        else:
            selected = select_frames_uniform_before_query(
                frames_by_identity=frames_by_identity,
                identity=identity,
                query_day_num=q_day,
                query_time_int=q_time,
                max_frames=plan["max_frames"],
            )
        timestamps = [format_frame_timestamp(d, t) for d, t, _ in selected]
        frame_paths = [p for _, _, p in selected]

        if not frame_paths:
            return (
                {
                    "key_evidence": [],
                    "reasoning": "no frames available in the requested window",
                    "summary": "observer received zero frames; need to broaden the search",
                },
                "",
                [],
                0.0,
            )

        prompt = build_observe_prompt(
            question=ctx["question"],
            query_time=ctx["query_time"],
            focus=plan["focus"],
            frame_timestamps=timestamps,
        )
        t0 = time.time()
        raw = call_gemini_multimodal(
            self.client,
            self.observer_model,
            prompt,
            frame_paths=frame_paths,
            frame_size=plan["frame_size"],
        )
        latency = round(time.time() - t0, 3)
        parsed = parse_json_response(raw) or {}
        if not isinstance(parsed.get("key_evidence"), list):
            parsed["key_evidence"] = []
        parsed.setdefault("reasoning", "")
        parsed.setdefault("summary", "")
        return parsed, raw, frame_paths, latency

    def _reflect(self, ctx: Dict) -> Tuple[Dict, str, float]:
        prompt = build_reflect_prompt(
            question=ctx["question"],
            options=ctx["options"],
            evidence=ctx["evidence"],
            rounds_used=ctx["round_idx"] + 1,
            max_rounds=self.max_rounds,
        )
        t0 = time.time()
        raw = call_gemini_text(self.client, self.reflector_model, prompt)
        latency = round(time.time() - t0, 3)
        parsed = parse_json_response(raw) or {}
        sufficient = bool(parsed.get("sufficient", False))
        confidence = parsed.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return (
            {
                "sufficient": sufficient,
                "confidence": confidence,
                "missing": str(parsed.get("missing") or ""),
                "rationale": str(parsed.get("rationale") or ""),
            },
            raw,
            latency,
        )

    def _synthesize(self, ctx: Dict) -> Tuple[str, str, float]:
        prompt = build_synthesize_prompt(
            question=ctx["question"],
            options=ctx["options"],
            evidence=ctx["evidence"],
        )
        t0 = time.time()
        raw = call_gemini_text(self.client, self.synthesizer_model, prompt)
        latency = round(time.time() - t0, 3)
        parsed = parse_json_response(raw) or {}
        valid = set(_validate_options_letters(ctx["options"]))
        sel = str(parsed.get("selected_option") or "").strip().upper()
        if sel in valid:
            return sel, raw, latency
        return "", raw, latency

    def run(
        self,
        *,
        example: Dict,
        frames_by_identity: Dict,
    ) -> Dict:
        identity = example.get("identity")
        query_time_str = example.get("query_time", "")
        query_day_num, query_time_int = parse_query_time(query_time_str)
        total_frames = count_frames_before_query(
            frames_by_identity, identity, query_day_num, query_time_int
        )
        days_spanned = days_spanned_before_query(
            frames_by_identity, identity, query_day_num, query_time_int
        )

        ctx = {
            "question": example.get("question") or example.get("query") or "",
            "options": example.get("options"),
            "query_type": example.get("query_type") or "unknown",
            "query_time": query_time_str,
            "query_day_num": query_day_num,
            "query_time_int": query_time_int,
            "identity": identity,
            "total_frames": total_frames,
            "days_spanned": days_spanned,
            "evidence": [],
            "missing_hint": None,
            "round_idx": 0,
            "remaining_budget": self.total_frame_budget,
        }

        trace: List[Dict] = []
        used_rounds = 0
        frames_used_total = 0

        for round_idx in range(self.max_rounds):
            ctx["round_idx"] = round_idx
            ctx["remaining_budget"] = max(0, self.total_frame_budget - frames_used_total)
            if ctx["remaining_budget"] < MIN_FRAMES_PER_OBSERVATION:
                # No budget left for another observation; stop and synthesize.
                break

            plan, plan_raw, plan_latency = self._plan(ctx)
            observation, obs_raw, frame_paths, obs_latency = self._observe(
                plan, ctx, frames_by_identity
            )
            frames_used_total += len(frame_paths)
            ctx["evidence"].append(observation)
            reflection, refl_raw, refl_latency = self._reflect(ctx)

            trace.append(
                {
                    "round": round_idx + 1,
                    "plan": plan,
                    "plan_raw": plan_raw,
                    "plan_latency_sec": plan_latency,
                    "n_frames_used": len(frame_paths),
                    "frames_used_total_after_round": frames_used_total,
                    "remaining_budget_after_round": max(
                        0, self.total_frame_budget - frames_used_total
                    ),
                    "observation": observation,
                    "observation_raw": obs_raw,
                    "observation_latency_sec": obs_latency,
                    "reflection": reflection,
                    "reflection_raw": refl_raw,
                    "reflection_latency_sec": refl_latency,
                }
            )
            used_rounds = round_idx + 1
            ctx["missing_hint"] = reflection.get("missing") or None

            if reflection.get("sufficient"):
                break

        pred, synth_raw, synth_latency = self._synthesize(ctx)

        return {
            "pred": pred,
            "synthesizer_raw": synth_raw,
            "synthesizer_latency_sec": synth_latency,
            "rounds_used": used_rounds,
            "total_frames_available": total_frames,
            "frames_used_total": frames_used_total,
            "total_frame_budget": self.total_frame_budget,
            "days_spanned": days_spanned,
            "agent_trace": trace,
        }
