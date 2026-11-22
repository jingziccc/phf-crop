#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs/model_outputs/final_freeze_20260713/mainline/llava"
DIRECT_KEY = ROOT / "outputs/model_outputs/mainline/direct_key_tasks_20260714/llava"
OUT = ROOT / "results/tables/analysis"
TASKS = ("aokvqa", "docvqa", "gqa", "pope", "textvqa", "vqav2", "vstar", "hrbench_4k", "hrbench_8k")


def result_path(task: str) -> Path:
    if task in {"vstar", "hrbench_4k", "hrbench_8k"}:
        directory = DIRECT_KEY / task
    elif task == "textvqa":
        directory = FINAL / "textvqa_full5000"
    else:
        directory = FINAL / task
    matches = [
        path for path in directory.glob(f"llava-{task}-*.json")
        if path.name not in {"evaluation_report.json", "run_manifest.json"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one result under {directory}, got {matches}")
    return matches[0]


def image_path(task: str, row: dict) -> Path:
    path = Path(str(row["image_path"]))
    return path if path.is_absolute() else ROOT / "data" / task / "images" / path


def main() -> None:
    cache: dict[Path, tuple[int, int]] = {}
    output = []
    for task in TASKS:
        rows = json.loads(result_path(task).read_text(encoding="utf-8"))
        max_sides = []
        for row in rows:
            path = image_path(task, row).resolve()
            if path not in cache:
                with Image.open(path) as image:
                    cache[path] = (int(image.width), int(image.height))
            max_sides.append(max(cache[path]))
        above = sum(value > 1024 for value in max_sides)
        sorted_sides = sorted(max_sides)
        median = sorted_sides[len(sorted_sides) // 2]
        high_share = 100 * above / len(rows)
        output.append(
            {
                "task": task,
                "samples": len(rows),
                "unique_images": len({image_path(task, row).resolve() for row in rows}),
                "median_max_side": median,
                "max_side_gt_1024": above,
                "max_side_gt_1024_pct": high_share,
                "resolution_group": "High-resolution" if high_share > 50 else "Standard-resolution",
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "dataset_resolution_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    lines = [
        "# Dataset Resolution Inventory",
        "",
        "A dataset is marked high-resolution when more than 50% of evaluated samples have original-image max side greater than 1024 pixels.",
        "",
        "| Task | N | Unique images | Median max side | >1024 N | >1024 share | Group |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in output:
        lines.append(
            f"| {row['task']} | {row['samples']} | {row['unique_images']} | {row['median_max_side']} | "
            f"{row['max_side_gt_1024']} | {row['max_side_gt_1024_pct']:.1f}% | {row['resolution_group']} |"
        )
    (OUT / "dataset_resolution_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
