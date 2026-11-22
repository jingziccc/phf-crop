#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "code/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import get_score  # noqa: E402


FINAL_ROOT = ROOT / "outputs/model_outputs/final_freeze_20260713/mainline"
DATA = ROOT / "data/textvqa/data.json"
CALIBRATION = ROOT / "outputs/provenance/final_freeze_20260713/textvqa_topmass_calibration_500.json"
HELDOUT = ROOT / "outputs/provenance/final_freeze_20260713/textvqa_paper_eval_heldout.json"
FROZEN_CONFIG = ROOT / "code/top1_hotspot/frozen_config.json"
SOURCES = {
    "llava": {
        "calibration": ROOT / "outputs/model_outputs/mainline/direct_key_tasks_20260714/llava/textvqa500/run_manifest.json",
        "heldout": FINAL_ROOT / "llava/textvqa/run_manifest.json",
    },
    "qwen2_5": {
        "calibration": ROOT / "outputs/provenance/final_freeze_20260713/calibration/qwen2_5/tm0p55/run_manifest.json",
        "heldout": FINAL_ROOT / "qwen2_5/textvqa/run_manifest.json",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_source(manifest_path: Path, expected_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"incomplete source manifest: {manifest_path}")
    if manifest.get("count") != expected_count or manifest.get("completed_count") != expected_count:
        raise RuntimeError(f"unexpected source count in {manifest_path}")
    output = Path(str(manifest["output"]))
    if not output.is_absolute():
        output = ROOT / output
    rows = load_json(output)
    if len(rows) != expected_count:
        raise RuntimeError(f"prediction count mismatch: {output}")
    return manifest, rows


def validate_partition() -> tuple[list[dict[str, Any]], set[str], set[str]]:
    full_rows = load_json(DATA)
    calibration_ids = {str(row["id"]) for row in load_json(CALIBRATION)}
    heldout_ids = {str(row["id"]) for row in load_json(HELDOUT)}
    full_ids = {str(row["id"]) for row in full_rows}
    if len(full_rows) != 5000 or len(full_ids) != 5000:
        raise RuntimeError("TextVQA data.json must contain 5,000 unique IDs")
    if len(calibration_ids) != 500 or len(heldout_ids) != 4500:
        raise RuntimeError("TextVQA split sizes must be 500 calibration + 4,500 held-out")
    if calibration_ids & heldout_ids or calibration_ids | heldout_ids != full_ids:
        raise RuntimeError("TextVQA calibration/held-out IDs do not form an exact partition")
    return full_rows, calibration_ids, heldout_ids


def merge_model(model: str, full_rows: list[dict[str, Any]], calibration_ids: set[str], heldout_ids: set[str]) -> None:
    cal_manifest_path = SOURCES[model]["calibration"]
    held_manifest_path = SOURCES[model]["heldout"]
    cal_manifest, cal_rows = complete_source(cal_manifest_path, 500)
    held_manifest, held_rows = complete_source(held_manifest_path, 4500)
    if float(cal_manifest.get("topmass", -1)) != 0.55 or float(held_manifest.get("topmass", -1)) != 0.55:
        raise RuntimeError(f"{model} TextVQA sources must both use topmass=0.55")

    cal_by_id = {str(row["id"]): row for row in cal_rows}
    held_by_id = {str(row["id"]): row for row in held_rows}
    if set(cal_by_id) != calibration_ids:
        raise RuntimeError(f"{model} calibration prediction IDs do not match the frozen calibration split")
    if set(held_by_id) != heldout_ids:
        raise RuntimeError(f"{model} held-out prediction IDs do not match the frozen held-out split")

    merged_by_id = {**cal_by_id, **held_by_id}
    merged: list[dict[str, Any]] = []
    for source in full_rows:
        sample_id = str(source["id"])
        prediction = merged_by_id[sample_id]
        if str(prediction.get("question", "")).strip() != str(source.get("question", "")).strip():
            raise RuntimeError(f"question mismatch for {model} TextVQA ID {sample_id}")
        merged.append(prediction)

    output_dir = FINAL_ROOT / model / "textvqa_full5000"
    output_dir.mkdir(parents=True, exist_ok=True)
    method = str(held_manifest["method"])
    output = output_dir / f"{model}-textvqa-{method}.json"
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw, crop = (float(value) for value in get_score.evaluate_textvqa(merged))
    report = [{
        "model_name": model,
        "task": "textvqa",
        "method": method,
        "raw_acc": raw,
        "crop_acc": crop,
    }]
    report_json = output_dir / "evaluation_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "evaluation_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(report)

    manifest = {
        "status": "complete",
        "artifact_type": "merged_prediction",
        "model": model,
        "task": "textvqa",
        "method": method,
        "topmass": 0.55,
        "count": 5000,
        "completed_count": 5000,
        "output": str(output),
        "evaluation_report": str(report_json),
        "merge_key": "original TextVQA id",
        "dataset": str(DATA),
        "partition": {"calibration": 500, "heldout": 4500},
        "source_manifests": [str(cal_manifest_path), str(held_manifest_path)],
        "source_outputs": [str(cal_manifest["output"]), str(held_manifest["output"])],
        "source_frozen_config_sha256": [
            cal_manifest.get("frozen_config_sha256"), held_manifest.get("frozen_config_sha256")
        ],
        "current_frozen_config_sha256": sha256(FROZEN_CONFIG),
        "scorer": "code/runtime/get_score.py:evaluate_textvqa",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[complete] {model} TextVQA5000 raw={raw:.4f} crop={crop:.4f} output={output}")


def main() -> None:
    full_rows, calibration_ids, heldout_ids = validate_partition()
    for model in SOURCES:
        merge_model(model, full_rows, calibration_ids, heldout_ids)


if __name__ == "__main__":
    main()
