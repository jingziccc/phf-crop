"""Replay Qwen VSTAR crop sizes with random whole-image positions."""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "scripts"), str(CODE / "tools")]

from run_top1_hotspot import image_path_for_row, load_model_adapter
from score_hrbench_official_style import normalize_mcq_letter

SEEDS = (4108, 4109, 4110)


def random_schedule(trace, image_size, seed, sample_id):
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    width, height = image_size
    boxes = []
    for stage in trace:
        x1, y1, x2, y2 = stage["crop_bbox"]
        crop_width, crop_height = x2 - x1, y2 - y1
        if not (0 < crop_width <= width and 0 < crop_height <= height):
            raise ValueError(f"invalid crop dimensions for sample {sample_id}")
        left = rng.randrange(width - crop_width + 1)
        top = rng.randrange(height - crop_height + 1)
        boxes.append([left, top, left + crop_width, top + crop_height])
    return boxes


def score_vstar(answer, row):
    letter = normalize_mcq_letter(answer)
    if not letter:
        text = answer.strip().rstrip(".").casefold()
        choices = re.findall(r"^\(([ABCD])\) (.+)$", row["question"], re.M)
        matches = [key for key, option in choices if option.strip().rstrip(".").casefold() == text]
        if len(matches) == 1:
            letter = matches[0]
    return int(letter == row["labels"])


def run_case(row, image, adapter):
    import torch

    sample_id = str(row["id"])
    trace = row["qpes_multistage_trace"]
    final_box = row["qpes_final_bbox"]
    if not trace or trace[-1]["crop_bbox"] != final_box:
        raise ValueError(f"final crop does not match the saved trace for sample {sample_id}")
    prompt = adapter.answer_prompt(row["question"])

    def answer(box):
        torch.manual_seed(4108 + int(sample_id))
        with torch.inference_mode():
            if box is None:
                return adapter.original_answer_fn(image, prompt, adapter.model, adapter.processor, 20)
            return adapter.crop_answer_fn(image, image.crop(box), prompt, adapter.model, adapter.processor, 20)

    nocrop_answer, phf_answer = answer(None), answer(final_box)
    random_runs = []
    for seed in SEEDS:
        boxes = random_schedule(trace, image.size, seed, sample_id)
        prediction = answer(boxes[-1])
        random_runs.append({"seed": seed, "answer": prediction,
                            "score": score_vstar(prediction, row), "crop_schedule": boxes})
    return {
        "id": sample_id,
        "nocrop_answer": nocrop_answer,
        "nocrop_score": score_vstar(nocrop_answer, row),
        "phf_answer": phf_answer,
        "phf_score": score_vstar(phf_answer, row),
        "random_mean_score": sum(run["score"] for run in random_runs) / len(random_runs),
        "random_runs": random_runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Saved Qwen PHF VSTAR predictions")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    rows = json.loads(args.reference.read_text(encoding="utf-8"))
    ids = {str(row["id"]) for row in rows}
    if not rows or len(ids) != len(rows):
        parser.error("reference must contain nonempty, unique VSTAR question IDs")
    adapter = load_model_adapter("qwen2_5", args.device)
    results = []
    for row in rows:
        with Image.open(image_path_for_row("vstar", row, str(args.image_root))) as image:
            results.append(run_case(row, image.convert("RGB"), adapter))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
