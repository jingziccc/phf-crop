#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOTS = (
    REPO / "outputs" / "llava_qwen_topmass_textvqa_vstar_hrbench_20260623",
    REPO / "outputs" / "blip_selected_param_vstar_hrbench_20260623",
)
DEFAULT_OUTPUT_DIR = REPO / "results" / "top1_hotspot_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute HRBench official-style cycle/category scores from existing "
            "PHF-Crop prediction JSON files. This keeps model inference fixed and "
            "uses deterministic MCQ letter matching in place of the official LLM judge."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=Path,
        default=None,
        help="Prediction JSON. Can be passed multiple times. Defaults to known HRBench output roots.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=None,
        help="Directory to scan recursively for *hrbench*.json prediction files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_predictions(args: argparse.Namespace) -> list[Path]:
    if args.prediction:
        return [path for path in args.prediction if path.exists()]

    roots = args.search_root or list(DEFAULT_SEARCH_ROOTS)
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*hrbench*.json"):
            if "_shards" in path.parts:
                continue
            name = path.name
            if name == "run_manifest.json" or name.endswith(".manifest.json"):
                continue
            if "evaluation_report" in name or "run_manifest" in name:
                continue
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def normalize_mcq_letter(answer: Any) -> str:
    if not isinstance(answer, str):
        return ""
    text = answer.strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return ""


def string_similarity(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()


def candidates_from_row(row: dict[str, Any]) -> list[str]:
    choices = row.get("mcq_choices")
    if isinstance(choices, dict):
        return [str(choices[key]) for key in ("A", "B", "C", "D") if key in choices]
    return str(row.get("question", "")).split("\n")[1:-1]


def predicted_letter(row: dict[str, Any], answer_key: str) -> str:
    answer = row.get(answer_key, "")
    letter = normalize_mcq_letter(answer)
    if letter in {"A", "B", "C", "D"}:
        return letter

    candidates = candidates_from_row(row)
    if not candidates:
        return ""
    best_index = max(
        range(len(candidates)),
        key=lambda index: string_similarity(str(answer), candidates[index]),
    )
    return "ABCD"[best_index]


def score_row(row: dict[str, Any], answer_key: str) -> float:
    pred = predicted_letter(row, answer_key)
    label = str(row.get("labels", "")).strip().upper()
    return 1.0 if pred == label else 0.0


def score_type(category: Any) -> str:
    if category == "single":
        return "FSP"
    if category == "cross":
        return "FCP"
    return str(category)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def infer_model(path: Path) -> str:
    name = path.name
    if "-" in name:
        return name.split("-", 1)[0]
    return "unknown"


def infer_task(path: Path, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        split = row.get("source_split")
        if split in {"hrbench_4k", "hrbench_8k"}:
            return str(split)
    raw = str(path)
    if "hrbench_4k" in raw:
        return "hrbench_4k"
    if "hrbench_8k" in raw:
        return "hrbench_8k"
    return "hrbench"


def infer_method(path: Path, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        method = row.get("crop_method")
        if method:
            return str(method)
    stem = path.stem
    parts = stem.split("-", 2)
    return parts[2] if len(parts) == 3 else stem


def official_cycle_scores(rows: list[dict[str, Any]], answer_key: str) -> dict[str, Any]:
    cycle_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sample_scores: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        cycle = str(row.get("cycle_category", "0"))
        typ = score_type(row.get("category"))
        score = score_row(row, answer_key)
        cycle_scores[cycle]["all"].append(score)
        cycle_scores[cycle][typ].append(score)
        sample_scores["all"].append(score)
        sample_scores[typ].append(score)

    cycle_rows: list[dict[str, Any]] = []
    cycle_avg: dict[str, float] = {}
    types = sorted(
        {typ for per_cycle in cycle_scores.values() for typ in per_cycle},
        key=lambda value: {"all": 0, "FSP": 1, "FCP": 2}.get(value, 99),
    )
    for cycle in sorted(cycle_scores, key=lambda value: (len(value), value)):
        for typ in types:
            values = cycle_scores[cycle].get(typ, [])
            if not values:
                continue
            cycle_rows.append(
                {
                    "cycle": cycle,
                    "type": typ,
                    "count": len(values),
                    "acc": 100.0 * mean(values),
                }
            )

    for typ in types:
        per_cycle = [
            100.0 * mean(cycle_scores[cycle][typ])
            for cycle in sorted(cycle_scores, key=lambda value: (len(value), value))
            if cycle_scores[cycle].get(typ)
        ]
        cycle_avg[typ] = mean(per_cycle)

    sample_avg = {
        typ: 100.0 * mean(values)
        for typ, values in sample_scores.items()
    }
    return {
        "cycle_rows": cycle_rows,
        "cycle_avg": cycle_avg,
        "sample_avg": sample_avg,
    }


def score_prediction(path: Path) -> dict[str, Any] | None:
    rows = load_json(path)
    if not isinstance(rows, list) or not rows:
        return None
    if not all(str(row.get("source_split", "")).startswith("hrbench") or "hrbench" in str(path) for row in rows):
        return None

    raw_scores = official_cycle_scores(rows, "original_answer")
    crop_scores = official_cycle_scores(rows, "crop_answer")
    return {
        "model": infer_model(path),
        "task": infer_task(path, rows),
        "method": infer_method(path, rows),
        "path": str(path),
        "count": len(rows),
        "raw": raw_scores,
        "crop": crop_scores,
    }


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    raw = report["raw"]["cycle_avg"]
    crop = report["crop"]["cycle_avg"]
    sample_raw = report["raw"]["sample_avg"]
    sample_crop = report["crop"]["sample_avg"]
    return {
        "model": report["model"],
        "task": report["task"],
        "method": report["method"],
        "count": report["count"],
        "raw_all": raw.get("all"),
        "crop_all": crop.get("all"),
        "raw_FSP": raw.get("FSP"),
        "crop_FSP": crop.get("FSP"),
        "raw_FCP": raw.get("FCP"),
        "crop_FCP": crop.get("FCP"),
        "sample_raw_all": sample_raw.get("all"),
        "sample_crop_all": sample_crop.get("all"),
        "sample_raw_FSP": sample_raw.get("FSP"),
        "sample_crop_FSP": sample_crop.get("FSP"),
        "sample_raw_FCP": sample_raw.get("FCP"),
        "sample_crop_FCP": sample_crop.get("FCP"),
        "source": report["path"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    reports = [
        report
        for path in discover_predictions(args)
        if (report := score_prediction(path)) is not None
    ]
    reports.sort(key=lambda item: (item["model"], item["task"], item["method"]))
    if not reports:
        raise RuntimeError("no HRBench prediction JSON files found")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hrbench_official_style_report.json"
    summary_path = output_dir / "hrbench_official_style_summary.csv"
    by_cycle_path = output_dir / "hrbench_official_style_by_cycle.csv"

    payload = {
        "metric_name": "hrbench_official_style_letter_match",
        "scorer_note": (
            "Uses HRBench official cycle/category aggregation. Per-sample scores are "
            "deterministic MCQ letter matches against labels, not the official LLM judge."
        ),
        "category_mapping": {"single": "FSP", "cross": "FCP"},
        "reports": reports,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = [summary_row(report) for report in reports]
    summary_fields = [
        "model",
        "task",
        "method",
        "count",
        "raw_all",
        "crop_all",
        "raw_FSP",
        "crop_FSP",
        "raw_FCP",
        "crop_FCP",
        "sample_raw_all",
        "sample_crop_all",
        "sample_raw_FSP",
        "sample_crop_FSP",
        "sample_raw_FCP",
        "sample_crop_FCP",
        "source",
    ]
    write_csv(summary_path, summary_rows, summary_fields)

    by_cycle_rows: list[dict[str, Any]] = []
    for report in reports:
        for answer_kind in ("raw", "crop"):
            for row in report[answer_kind]["cycle_rows"]:
                by_cycle_rows.append(
                    {
                        "model": report["model"],
                        "task": report["task"],
                        "method": report["method"],
                        "answer_kind": answer_kind,
                        **row,
                        "source": report["path"],
                    }
                )
    write_csv(
        by_cycle_path,
        by_cycle_rows,
        ["model", "task", "method", "answer_kind", "cycle", "type", "count", "acc", "source"],
    )

    print(f"[wrote] {json_path}")
    print(f"[wrote] {summary_path}")
    print(f"[wrote] {by_cycle_path}")
    for row in summary_rows:
        print(
            f"{row['model']} {row['task']} {row['method']} "
            f"raw={row['raw_all']:.2f} crop={row['crop_all']:.2f} "
            f"FSP={row['crop_FSP']:.2f} FCP={row['crop_FCP']:.2f}"
        )


if __name__ == "__main__":
    main()
