"""
Multiscale episodic memory generation.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm

from ...llm import LLMModel
from .gen_multiscale import gen_multiscale


def _process_video_dir(
    video_dir: str,
    model_name: str,
    base_name: str,
    windows: List[int],
    granularity_names: List[str],
    perspective: str,
) -> tuple[str, str]:
    base_json = os.path.join(video_dir, base_name)
    if not os.path.exists(base_json):
        return video_dir, f"Skipping {video_dir}: {base_name} not found"

    llm = LLMModel(model_name=model_name)
    gen_multiscale(
        input_json=base_json,
        save_dir=video_dir,
        llm=llm,
        windows=windows,
        granularity_names=granularity_names,
        perspective=perspective,
    )
    return video_dir, f"[{os.path.basename(video_dir)}] Multiscale captions complete"


def generate_multiscale_memory(
    caption_dir: str,
    model_name: str = "gpt-5-mini",
    base_name: str = "10sec.json",
    windows: Optional[List[int]] = None,
    granularity_names: Optional[List[str]] = None,
    perspective: str = "general",
):
    """
    Generate multiscale captions.

    If caption_dir contains a file named base_name directly, it is treated
    as a single video directory. Otherwise, each subdirectory of caption_dir
    is treated as a separate video directory.

    Args:
        caption_dir: Directory (single video) or parent of {videoID}/ subdirs.
        model_name: LLM model name for summarization.
        base_name: Filename of the base caption JSON inside each directory.
        windows: Time window sizes in seconds for each level.
        granularity_names: Output granularity names for each level.
        perspective: "egocentric" or "general" prompt style.
    """
    windows = windows or [30, 180, 600]
    granularity_names = granularity_names or ["30sec", "3min", "10min"]

    base_in_root = os.path.exists(os.path.join(caption_dir, base_name))
    if base_in_root:
        video_dirs = [caption_dir]
    else:
        video_dirs = sorted(
            os.path.join(caption_dir, d)
            for d in os.listdir(caption_dir)
            if os.path.isdir(os.path.join(caption_dir, d))
        )

    if not video_dirs:
        print(f"No video directories found in {caption_dir}")
        return

    print(f"Processing {len(video_dirs)} video(s) for multiscale memory (perspective={perspective})...")
    failures: list[tuple[str, str]] = []
    max_workers = min(8, len(video_dirs)) if video_dirs else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video_dir = {
            executor.submit(
                _process_video_dir,
                video_dir,
                model_name,
                base_name,
                windows,
                granularity_names,
                perspective,
            ): video_dir
            for video_dir in video_dirs
        }
        progress_bar = tqdm(
            as_completed(future_to_video_dir),
            total=len(future_to_video_dir),
            desc="Multiscale memory",
            unit="video",
        )
        for future in progress_bar:
            video_dir = future_to_video_dir[future]
            progress_bar.set_postfix(video=os.path.basename(video_dir))
            try:
                _, message = future.result()
                tqdm.write(message)
            except Exception as exc:
                failures.append((video_dir, str(exc)))
                tqdm.write(f"[{os.path.basename(video_dir)}] Failed: {exc}")
        progress_bar.close()

    if failures:
        failed_videos = ", ".join(os.path.basename(video_dir) for video_dir, _ in failures)
        raise RuntimeError(f"Multiscale memory generation failed for {len(failures)} video(s): {failed_videos}")

    print("Multiscale memory generation complete.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Generate multiscale episodic memory.")
    parser.add_argument("--caption_dir", required=True, help="Single video dir or parent of {videoID}/ subdirs")
    parser.add_argument("--model", default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--base_name", default="10sec.json", help="Base caption filename in each directory")
    parser.add_argument("--windows", default="30,180,600", help="Comma-separated window sizes in seconds")
    parser.add_argument("--granularity_names", default="30sec,3min,10min", help="Comma-separated output granularity names")
    parser.add_argument("--perspective", default="general", choices=["egocentric", "general"], help="Prompt style: egocentric (first-person) or general (third-person)")

    args = parser.parse_args()

    generate_multiscale_memory(
        caption_dir=args.caption_dir,
        model_name=args.model,
        base_name=args.base_name,
        windows=[int(w) for w in args.windows.split(",")],
        granularity_names=args.granularity_names.split(","),
        perspective=args.perspective,
    )
