#!/usr/bin/env python3
"""Evaluate PHF-Crop boxes against VSTAR and TextVQA ground-truth boxes.

This is metadata-only analysis: it reads saved coordinates and image dimensions
but never decodes or displays image pixels.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "code" / "runtime"
sys.path.insert(0, str(RUNTIME))

import get_score  # noqa: E402


BLIP_MAINLINE = ROOT / "outputs/model_outputs/mainline/blip_subtraction_tile1024_stop1"
FINAL_ROOT = ROOT / "outputs/model_outputs/final_freeze_20260713"
HELDOUT_TEXTVQA = (
    ROOT / "outputs/provenance/final_freeze_20260713/textvqa_paper_eval_heldout.json"
)
OUT_DIR = ROOT / "results/tables/final_freeze_20260713"
MODELS = ("llava", "blip", "qwen2_5")
MODEL_LABELS = {
    "llava": "LLaVA",
    "blip": "InstructBLIP",
    "qwen2_5": "Qwen2.5-VL",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_path(model: str, task: str) -> Path:
    if model in {"llava", "qwen2_5"}:
        directory = FINAL_ROOT / "mainline" / model / task
    elif task == "vstar":
        directory = BLIP_MAINLINE / "highres" / task
    elif task == "textvqa":
        directory = BLIP_MAINLINE / "standard" / task
    else:
        raise KeyError(task)
    matches = [
        path
        for path in directory.glob(f"{model}-{task}-*.json")
        if path.name not in {"evaluation_report.json", "run_manifest.json"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction JSON under {directory}, got {matches}")
    return matches[0]


def xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid XYWH box: {box}")
    return x, y, x + width, y + height


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )


def pair_metrics(crop: Iterable[float], target: Iterable[float]) -> dict[str, float]:
    crop = tuple(float(value) for value in crop)
    target = tuple(float(value) for value in target)
    inter = intersection_area(crop, target)
    crop_area = box_area(crop)
    target_area = box_area(target)
    union = crop_area + target_area - inter
    tx1, ty1, tx2, ty2 = target
    cx, cy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    px1, py1, px2, py2 = crop
    return {
        "iou": inter / union if union else 0.0,
        "target_recall": inter / target_area if target_area else 0.0,
        "crop_precision": inter / crop_area if crop_area else 0.0,
        "center_hit": float(px1 <= cx <= px2 and py1 <= cy <= py2),
    }


def aggregate_target_metrics(
    crop: Iterable[float], targets: list[tuple[float, float, float, float]]
) -> dict[str, float]:
    pairs = [pair_metrics(crop, target) for target in targets]
    return {
        "mean_iou": fmean(pair["iou"] for pair in pairs),
        "best_iou": max(pair["iou"] for pair in pairs),
        "mean_target_recall": fmean(pair["target_recall"] for pair in pairs),
        "any_target_recall": max(pair["target_recall"] for pair in pairs),
        "mean_crop_precision": fmean(pair["crop_precision"] for pair in pairs),
        "any_center_hit": max(pair["center_hit"] for pair in pairs),
        "all_center_hit": min(pair["center_hit"] for pair in pairs),
    }


def vstar_annotations() -> dict[str, dict[str, Any]]:
    raw_root = ROOT / "data/vstar/raw/vstar_bench"
    result: dict[str, dict[str, Any]] = {}
    for path in raw_root.rglob("*.json"):
        row = load_json(path)
        if not isinstance(row, dict) or "bbox" not in row or "question" not in row:
            continue
        key = str(path.relative_to(raw_root).with_suffix(""))
        if key in result:
            raise RuntimeError(f"duplicate VSTAR annotation key: {key}")
        result[key] = {**row, "annotation_path": str(path)}
    return result


def dataset_metadata(task: str) -> dict[str, dict[str, Any]]:
    if task == "vstar":
        annotations = vstar_annotations()
        result = {}
        for row in load_json(ROOT / "data/vstar/data.json"):
            key = str(Path(row["image_path"]).with_suffix(""))
            source = annotations.get(key)
            if source is None:
                raise RuntimeError(f"missing VSTAR annotation for {row['image_path']}")
            if source["question"].strip() != row["short_question"].strip():
                raise RuntimeError(f"VSTAR question mismatch for {row['id']}")
            targets = [xywh_to_xyxy(box) for box in source["bbox"]]
            result[str(row["id"])] = {
                "targets": targets,
                "source": source["annotation_path"],
                "target_objects": source.get("target_object", []),
            }
        return result

    if task == "textvqa":
        heldout_ids = {str(row["id"]) for row in load_json(HELDOUT_TEXTVQA)}
        result = {}
        for row in load_json(ROOT / "data/textvqa/data_with_bbox.json"):
            if not row.get("has_bbox") or str(row["id"]) not in heldout_ids:
                continue
            result[str(row["id"])] = {
                "targets": [tuple(float(value) for value in row["bbox_xyxy"])],
                "source": "jrzhang/TextVQA_GT_bbox",
                "target_objects": [],
            }
        return result

    raise KeyError(task)


def image_size(task: str, image_path: str) -> tuple[int, int]:
    path = ROOT / "data" / task / "images" / image_path
    with Image.open(path) as image:
        return image.size


def answer_score(task: str, row: dict[str, Any]) -> float:
    if task == "textvqa":
        return float(get_score.get_acc(str(row["crop_answer"]), list(row["labels"])))
    if task == "vstar":
        candidates = str(row["question"]).split("\n")[1:-1]
        answer = str(row["crop_answer"])
        letter = get_score.normalize_mcq_letter(answer)
        predicted = letter if letter else "ABCD"[get_score.get_index(answer, candidates)]
        return 100.0 if predicted == str(row["labels"]).strip().upper() else 0.0
    raise KeyError(task)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "answer_score": mean(rows, "answer_score"),
        "gt_area_pct": 100.0 * mean(rows, "gt_area_fraction"),
        "crop_area_pct": 100.0 * mean(rows, "crop_area_fraction"),
        "mean_iou": mean(rows, "mean_iou"),
        "best_iou": mean(rows, "best_iou"),
        "mean_target_recall": mean(rows, "mean_target_recall"),
        "any_target_recall": mean(rows, "any_target_recall"),
        "mean_crop_precision": mean(rows, "mean_crop_precision"),
        "any_center_hit_pct": 100.0 * mean(rows, "any_center_hit"),
        "recall_ge_50_pct": 100.0
        * fmean(row["any_target_recall"] >= 0.5 for row in rows),
        "recall_ge_90_pct": 100.0
        * fmean(row["any_target_recall"] >= 0.9 for row in rows),
        "recall_ge_99_pct": 100.0
        * fmean(row["any_target_recall"] >= 0.99 for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    stage_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    coverage_rows = []

    for task in ("vstar", "textvqa"):
        metadata = dataset_metadata(task)
        packaged_count = (
            len(load_json(HELDOUT_TEXTVQA))
            if task == "textvqa"
            else len(load_json(ROOT / f"data/{task}/data.json"))
        )
        coverage_rows.append(
            {
                "dataset": task.upper() if task == "vstar" else "TextVQA",
                "packaged_n": packaged_count,
                "gt_box_n": len(metadata),
                "coverage_pct": 100.0 * len(metadata) / packaged_count,
                "mapping_key": "category/image stem + exact question"
                if task == "vstar"
                else "packaged row id / TextVQA validation order",
                "bbox_format": "source XYWH converted to XYXY"
                if task == "vstar"
                else "provided XYXY",
            }
        )
        for model in MODELS:
            prediction = prediction_path(model, task)
            records = load_json(prediction)
            if task == "textvqa":
                heldout_ids = {str(row["id"]) for row in load_json(HELDOUT_TEXTVQA)}
                records = [row for row in records if str(row["id"]) in heldout_ids]
            localized = []
            for record in records:
                meta = metadata.get(str(record["id"]))
                if meta is None:
                    continue
                targets = meta["targets"]
                width, height = image_size(task, str(record["image_path"]))
                image_area = float(width * height)
                crop = tuple(float(value) for value in record["qpes_final_bbox"])
                metrics = aggregate_target_metrics(crop, targets)
                item = {
                    **metrics,
                    "answer_score": answer_score(task, record),
                    "gt_area_fraction": fmean(box_area(target) for target in targets)
                    / image_area,
                    "crop_area_fraction": box_area(crop) / image_area,
                }
                localized.append(item)

                for stage in record.get("qpes_multistage_trace", []):
                    stage_crop = stage.get("crop_bbox")
                    if not stage_crop:
                        continue
                    stage_groups[(task, model, int(stage["stage"]))].append(
                        {
                            **aggregate_target_metrics(stage_crop, targets),
                            "crop_area_fraction": box_area(stage_crop) / image_area,
                        }
                    )

            if not localized:
                raise RuntimeError(f"no GT-localized rows for {model}/{task}")
            base = {
                "dataset": "VSTAR" if task == "vstar" else "TextVQA",
                "model": MODEL_LABELS[model],
                "prediction": str(prediction.relative_to(ROOT)),
            }
            summary_rows.append({**base, **summarize(localized)})
            for label, subset in (
                ("answer-correct", [row for row in localized if row["answer_score"] > 0]),
                ("answer-wrong", [row for row in localized if row["answer_score"] == 0]),
            ):
                if subset:
                    outcome_rows.append({**base, "outcome": label, **summarize(subset)})

    stage_rows = []
    for (task, model, stage), rows in sorted(stage_groups.items()):
        stage_rows.append(
            {
                "dataset": "VSTAR" if task == "vstar" else "TextVQA",
                "model": MODEL_LABELS[model],
                "stage": stage,
                "n": len(rows),
                "crop_area_pct": 100.0 * mean(rows, "crop_area_fraction"),
                "mean_iou": mean(rows, "mean_iou"),
                "any_target_recall": mean(rows, "any_target_recall"),
                "any_center_hit_pct": 100.0 * mean(rows, "any_center_hit"),
                "recall_ge_90_pct": 100.0
                * fmean(row["any_target_recall"] >= 0.9 for row in rows),
            }
        )

    summary_columns = [
        "dataset",
        "model",
        "n",
        "answer_score",
        "gt_area_pct",
        "crop_area_pct",
        "mean_iou",
        "any_target_recall",
        "mean_crop_precision",
        "any_center_hit_pct",
        "recall_ge_50_pct",
        "recall_ge_90_pct",
        "recall_ge_99_pct",
        "prediction",
    ]
    outcome_columns = [
        "dataset",
        "model",
        "outcome",
        "n",
        "mean_iou",
        "any_target_recall",
        "mean_crop_precision",
        "any_center_hit_pct",
        "recall_ge_90_pct",
    ]
    stage_columns = [
        "dataset",
        "model",
        "stage",
        "n",
        "crop_area_pct",
        "mean_iou",
        "any_target_recall",
        "any_center_hit_pct",
        "recall_ge_90_pct",
    ]
    coverage_columns = [
        "dataset",
        "packaged_n",
        "gt_box_n",
        "coverage_pct",
        "mapping_key",
        "bbox_format",
    ]
    write_csv(OUT_DIR / "gt_localization_summary.csv", summary_rows, summary_columns)
    write_csv(OUT_DIR / "gt_localization_by_outcome.csv", outcome_rows, outcome_columns)
    write_csv(OUT_DIR / "gt_localization_by_stage.csv", stage_rows, stage_columns)
    write_csv(OUT_DIR / "gt_box_mapping_coverage.csv", coverage_rows, coverage_columns)

    concise_summary_columns = [column for column in summary_columns if column != "prediction"]
    report = [
        "# Ground-Truth Localization Analysis",
        "",
        "VSTAR boxes come from the original per-image annotations and are XYWH; "
        "they are mapped by category/image stem and exact question, then converted "
        "to XYXY. TextVQA boxes come from the author-released "
        "`jrzhang/TextVQA_GT_bbox` data. Ground-truth boxes are analysis-only and "
        "are never used by PHF-Crop.",
        "",
        "IoU is intentionally complemented by target recall (fraction of the GT "
        "box covered), crop precision (fraction of the crop occupied by the GT), "
        "and center hit. For tiny targets, a useful context-preserving crop can "
        "have high target recall but low IoU.",
        "",
        "## Annotation coverage",
        "",
        markdown_table(coverage_rows, coverage_columns),
        "",
        "## Final selected crop",
        "",
        markdown_table(summary_rows, concise_summary_columns),
        "",
        "## Stratified by answer outcome",
        "",
        markdown_table(outcome_rows, outcome_columns),
        "",
        "## Stage-wise localization",
        "",
        "The stage table evaluates all saved candidate stages. The final table "
        "above evaluates the stage actually selected by adaptive stop.",
        "",
        markdown_table(stage_rows, stage_columns),
        "",
        "Raw per-sample records are intentionally not copied into the paper-facing "
        "result tree; the source prediction paths are retained in the CSV summary.",
    ]
    (OUT_DIR / "gt_localization_analysis.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
