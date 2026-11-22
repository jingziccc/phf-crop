#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
RUNTIME = CODE / "runtime"
SCRIPTS = CODE / "scripts"
for path in (CODE, RUNTIME, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_top1_hotspot import (  # noqa: E402
    calibration_manifest_sha256,
    image_path_for_row,
    pythonpath_env,
)
from top1_hotspot.adapters import load_model_adapter  # noqa: E402
from top1_hotspot.config import (  # noqa: E402
    COMMON_MIN_CROP_SIDE,
    FROZEN_CONFIG,
    METHOD_CN_NAME,
    METHOD_DISPLAY_NAME,
    METHOD_FULL_NAME,
    SUPPORTED_MODELS,
    frozen_config_sha256,
)
from top1_hotspot.geometry import bbox_to_int_tuple, dumps_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the matched first-stage-only answer ablation from a final PHF trace.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(args: argparse.Namespace, output_dir: Path) -> Path:
    return output_dir / f"{args.model}-{args.task}-phf_crop_first_stage_only.json"


def manifest_complete(output_dir: Path) -> bool:
    path = output_dir / "run_manifest.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("status") == "complete" and data.get("count") == data.get("completed_count")


def evaluate(output_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_env()
    subprocess.run(
        [sys.executable, str(RUNTIME / "get_score.py"), "--data_dir", str(output_dir), "--save_path", str(output_dir)],
        cwd=str(ROOT), env=env, check=True,
    )


def write_manifest(path: Path, *, args: argparse.Namespace, status: str, count: int, output: Path) -> None:
    data = {
        "status": status,
        "model": args.model,
        "task": args.task,
        "variant": "first_stage_only",
        "source_json": str(Path(args.source_json).resolve()),
        "output": str(output),
        "count": int(count),
        "completed_count": int(count if status == "complete" else 0),
        "frozen_config_sha256": frozen_config_sha256(),
        "calibration_manifest_sha256": calibration_manifest_sha256(),
        "min_crop_side": COMMON_MIN_CROP_SIDE,
        "final_bbox_semantics": "current_centered_square_crop",
    }
    path.write_text(dumps_json(data), encoding="utf-8")


def run_worker(args: argparse.Namespace) -> None:
    if args.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["QPES_BLIP_BG_SUB_ALPHA"] = str(FROZEN_CONFIG["models"]["blip"]["background_subtraction_alpha"])
    import torch

    source_rows = load_rows(Path(args.source_json))
    end = len(source_rows) if args.end_index is None else min(int(args.end_index), len(source_rows))
    selected = source_rows[int(args.start_index):end]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path(args, output_dir)
    existing = {str(row["id"]): row for row in load_rows(out_path)} if args.resume and out_path.is_file() else {}
    adapter = load_model_adapter(
        str(args.model),
        "cuda" if torch.cuda.is_available() else "cpu",
        min_crop_side=COMMON_MIN_CROP_SIDE,
        blip_bg_sub_alpha=float(FROZEN_CONFIG["models"]["blip"]["background_subtraction_alpha"]),
    )
    outputs: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row["id"])
        if sample_id in existing:
            outputs.append(existing[sample_id])
            continue
        trace = list(row.get("qpes_multistage_trace") or [])
        if not trace:
            raise RuntimeError(f"source row id={sample_id} lacks qpes_multistage_trace")
        first_stage = trace[0]
        bbox = bbox_to_int_tuple(first_stage["crop_bbox"])
        from PIL import Image
        image = Image.open(image_path_for_row(str(args.task), row, args.image_root)).convert("RGB")
        crop = image.crop(bbox)
        answer_prompt = adapter.answer_prompt(str(row["question"]))
        crop_answer = adapter.crop_answer_fn(
            image, crop, answer_prompt, adapter.model, adapter.processor, int(args.max_new_tokens)
        )
        out = dict(row)
        for key in (
            "qpes_adaptive_stop", "qpes_stop_reason", "qpes_target_labels", "qpes_target_sides",
            "qpes_multistage_ratios", "qpes_num_steps",
        ):
            out.pop(key, None)
        out.update({
            "crop_answer": crop_answer,
            "bbox": list(bbox),
            "crop_method": "phf_crop_first_stage_only",
            "method_display_name": METHOD_DISPLAY_NAME,
            "method_chinese_name": METHOD_CN_NAME,
            "method_full_name": METHOD_FULL_NAME,
            "qpes_ablation_variant": "first_stage_only",
            "qpes_num_steps": 1,
            "qpes_final_bbox": list(bbox),
            "qpes_final_bbox_semantics": "current_centered_square_crop",
            "qpes_multistage_trace": [first_stage],
            "qpes_frozen_config_sha256": frozen_config_sha256(),
            "qpes_calibration_manifest_sha256": calibration_manifest_sha256(),
        })
        outputs.append(out)
        out_path.write_text(dumps_json(outputs), encoding="utf-8")
    out_path.write_text(dumps_json(outputs), encoding="utf-8")
    if not args.no_evaluate:
        evaluate(output_dir)
    write_manifest(output_dir / "run_manifest.json", args=args, status="complete", count=len(outputs), output=out_path)


def split_ranges(total: int, count: int) -> list[tuple[int, int]]:
    count = max(1, min(count, total))
    return [(total * i // count, total * (i + 1) // count) for i in range(count)]


def run_parent(args: argparse.Namespace) -> None:
    gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    rows = load_rows(Path(args.source_json))
    output_dir = Path(args.output_dir)
    if args.resume and manifest_complete(output_dir):
        print(f"[skip] complete: {output_dir}")
        return
    shard_root = output_dir / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[Any], Path, Path, Any]] = []
    for index, ((start, end), gpu) in enumerate(zip(split_ranges(len(rows), len(gpus)), gpus)):
        shard_dir = shard_root / f"shard_{start:06d}_{end:06d}_gpu{gpu}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "run.log"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--model", args.model, "--task", args.task,
            "--source-json", args.source_json, "--output-dir", str(shard_dir),
            "--gpu", gpu, "--start-index", str(start), "--end-index", str(end),
            "--max-new-tokens", str(args.max_new_tokens), "--no-evaluate",
        ]
        if args.image_root:
            command.extend(["--image-root", args.image_root])
        if args.resume:
            command.append("--resume")
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((process, shard_dir, log_path, handle))
    for process, _shard_dir, log_path, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code != 0:
            raise RuntimeError(f"first-stage shard failed; see {log_path}")
    merged: dict[str, dict[str, Any]] = {}
    for _process, shard_dir, _log_path, _handle in processes:
        for row in load_rows(output_path(args, shard_dir)):
            sample_id = str(row["id"])
            if sample_id in merged:
                raise RuntimeError(f"duplicate merged id: {sample_id}")
            merged[sample_id] = row
    ordered = [merged[str(row["id"])] for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_path(args, output_dir)
    final_path.write_text(dumps_json(ordered), encoding="utf-8")
    evaluate(output_dir)
    write_manifest(output_dir / "run_manifest.json", args=args, status="complete", count=len(ordered), output=final_path)
    shutil.rmtree(shard_root)
    print(f"[complete] first-stage-only {args.model}/{args.task} count={len(ordered)}")


def main() -> None:
    args = parse_args()
    if args.gpus:
        run_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
