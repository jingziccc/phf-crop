#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from top1_hotspot.config import FROZEN_CONFIG_PATH  # noqa: E402


RUNNER = CODE / "scripts/run_top1_hotspot.py"
FIRST_STAGE = CODE / "scripts/run_first_stage_from_trace.py"
OUTPUT_ROOT = ROOT / "outputs/model_outputs/final_freeze_20260713"
ABLATION_ROOT = ROOT / "outputs/model_outputs/ablations"
PROVENANCE = ROOT / "outputs/provenance/final_freeze_20260713"
HELDOUT_TEXTVQA = PROVENANCE / "textvqa_paper_eval_heldout.json"
TASKS = (
    "aokvqa", "docvqa", "gqa", "pope", "textvqa", "vqav2",
    "vstar", "hrbench_4k", "hrbench_8k",
)
CORE_TASKS = ("vstar", "hrbench_4k", "hrbench_8k")
MODELS = ("llava", "qwen2_5")
PAPER_CAPS = {
    "docvqa": 5000,
    "gqa": 5000,
    "pope": 5000,
    "vqav2": 5000,
}
TABLE_BUILDER = CODE / "tools/build_final_freeze_tables.py"
LOCALIZATION = CODE / "tools/analyze_gt_localization.py"
LOCK_METHOD = CODE / "tools/lock_final_method.py"
FINAL_TABLES = ROOT / "results/tables/final_freeze_20260713"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked final PHF mainline and minimal core ablations.")
    parser.add_argument("--gpus", default="1,2,4")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--models", default=",".join(MODELS),
        help="Comma-separated model subset in canonical order; e.g. llava or llava,qwen2_5.",
    )
    parser.add_argument(
        "--tasks", default=",".join(TASKS),
        help="Comma-separated task subset in canonical order.",
    )
    parser.add_argument(
        "--skip-core-ablations", action="store_true",
        help="Run only the selected mainline tasks and skip Iterative-Local/First-stage-only jobs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the job matrix without running models.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "run_manifest.json"
    if not path.is_file():
        return None
    data = load_json(path)
    if data.get("status") != "complete" or data.get("count") != data.get("completed_count"):
        return None
    output = Path(str(data.get("output", "")))
    if not output.is_file():
        return None
    if len(load_json(output)) != int(data["count"]):
        return None
    return data


def gpu_memory() -> dict[str, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    values: dict[str, float] = {}
    for line in result.stdout.splitlines():
        index, memory = [item.strip() for item in line.split(",")]
        values[index] = float(memory)
    return values


def run_job(
    *, name: str, command: list[str], output_dir: Path, log_path: Path,
    env: dict[str, str], resume: bool, gpu_ids: list[str],
) -> dict[str, Any]:
    existing = complete_manifest(output_dir) if resume else None
    if existing is not None:
        runtime_record = output_dir / "runtime_record.json"
        prior = load_json(runtime_record) if runtime_record.is_file() else {}
        return {
            **prior,
            "name": name,
            "status": "skipped_complete",
            "output_dir": str(output_dir),
            "count": int(existing["count"]),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    baseline_memory = gpu_memory()
    peak_memory = {gpu: baseline_memory[gpu] for gpu in gpu_ids}
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command + (["--resume"] if resume else []),
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            current = gpu_memory()
            for gpu in gpu_ids:
                peak_memory[gpu] = max(peak_memory[gpu], current[gpu])
            time.sleep(5.0)
        return_code = int(process.returncode)
    manifest = complete_manifest(output_dir)
    record = {
        "name": name,
        "status": "complete" if return_code == 0 and manifest is not None else "failed",
        "return_code": return_code,
        "command": command,
        "output_dir": str(output_dir),
        "log": str(log_path),
        "elapsed_seconds": time.monotonic() - started,
        "count": int(manifest["count"]) if manifest is not None else None,
        "baseline_gpu_memory_mb": {gpu: baseline_memory[gpu] for gpu in gpu_ids},
        "peak_gpu_memory_mb": {gpu: peak_memory[gpu] for gpu in gpu_ids},
        "peak_increment_mb": {gpu: peak_memory[gpu] - baseline_memory[gpu] for gpu in gpu_ids},
    }
    if record["status"] != "complete":
        raise RuntimeError(f"final campaign job failed: {name}; see {log_path}")
    (output_dir / "runtime_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def prediction_from(output_dir: Path) -> Path:
    manifest = complete_manifest(output_dir)
    if manifest is None:
        raise RuntimeError(f"incomplete source output: {output_dir}")
    return Path(str(manifest["output"]))


def write_runtime_table(records: list[dict[str, Any]], suffix: str = "") -> None:
    measured = [row for row in records if row.get("elapsed_seconds") is not None]
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    columns = ["Job", "N", "Wall time (s)", "Samples/s", "Peak GPU increment (MB)"]
    rows = []
    for record in measured:
        count = int(record.get("count") or 0)
        elapsed = float(record["elapsed_seconds"])
        peak = max(float(value) for value in record["peak_increment_mb"].values())
        rows.append({
            "Job": record["name"], "N": count, "Wall time (s)": f"{elapsed:.1f}",
            "Samples/s": f"{count / elapsed:.4f}", "Peak GPU increment (MB)": f"{peak:.0f}",
        })
    import csv
    stem = "measured_runtime_gpu_memory" + (f"_{suffix}" if suffix else "")
    with (FINAL_TABLES / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Measured Runtime and GPU Memory", "",
        "Each row is one task sharded jointly across the configured GPUs.", "",
        "| " + " | ".join(columns) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    (FINAL_TABLES / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_models(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown --models entries: {unknown}; choices={MODELS}")
    if len(requested) != len(set(requested)):
        raise ValueError("--models contains duplicates")
    ordered = tuple(model for model in MODELS if model in requested)
    if not ordered:
        raise ValueError("--models must contain at least one model")
    return ordered


def selected_tasks(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown --tasks entries: {unknown}; choices={TASKS}")
    if len(requested) != len(set(requested)):
        raise ValueError("--tasks contains duplicates")
    ordered = tuple(task for task in TASKS if task in requested)
    if not ordered:
        raise ValueError("--tasks must contain at least one task")
    return ordered


def main() -> None:
    args = parse_args()
    config = load_json(FROZEN_CONFIG_PATH)
    if config.get("status") != "locked":
        raise RuntimeError("final campaign requires frozen_config.json status=locked")
    gpus = ",".join(item.strip() for item in str(args.gpus).split(",") if item.strip())
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    gpu_ids = gpus.split(",")
    models = selected_models(str(args.models))
    tasks = selected_tasks(str(args.tasks))
    core_tasks = () if args.skip_core_ablations else tuple(task for task in CORE_TASKS if task in tasks)
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    records: list[dict[str, Any]] = []
    full_scope = models == MODELS and tasks == TASKS and core_tasks == CORE_TASKS
    suffix = "" if full_scope else "_".join(models)
    status_path = PROVENANCE / ("final_campaign_status.json" if full_scope else f"final_campaign_status_{suffix}.json")

    if args.dry_run:
        plan = {
            "models": models,
            "tasks": tasks,
            "core_tasks": core_tasks,
            "gpus": gpu_ids,
            "mainline_jobs": [f"mainline_{model}_{task}" for model in models for task in tasks],
            "iterative_local_jobs": [f"iterative_local_{model}_{task}" for model in models for task in core_tasks],
            "first_stage_only_jobs": [f"first_stage_only_{model}_{task}" for model in models for task in core_tasks],
            "post_commands_deferred": not full_scope,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    try:
        for model in models:
            # Finish the complete LLaVA evidence package before starting any
            # formal Qwen run. Qwen is substantially slower, so this ordering
            # makes the first paper-ready model package available earlier.
            for task in tasks:
                output_dir = OUTPUT_ROOT / "mainline" / model / task
                command = [
                    str(args.python), str(RUNNER),
                    "--model", model, "--task", task,
                    "--gpus", gpus, "--output-dir", str(output_dir),
                    "--log-every", "25",
                ]
                if task == "textvqa":
                    command.extend(["--question-path", str(HELDOUT_TEXTVQA)])
                if task in PAPER_CAPS:
                    command.extend(["--cap", str(PAPER_CAPS[task])])
                records.append(run_job(
                    name=f"mainline_{model}_{task}", command=command, output_dir=output_dir,
                    log_path=PROVENANCE / "logs" / f"mainline_{model}_{task}.log",
                    env=env, resume=bool(args.resume), gpu_ids=gpu_ids,
                ))

            for task in core_tasks:
                output_dir = ABLATION_ROOT / "iterative_local" / model / task
                command = [
                    str(args.python), str(RUNNER),
                    "--model", model, "--task", task,
                    "--gpus", gpus, "--output-dir", str(output_dir),
                    "--localization-strategy", "iterative_local", "--log-every", "25",
                ]
                records.append(run_job(
                    name=f"iterative_local_{model}_{task}", command=command, output_dir=output_dir,
                    log_path=PROVENANCE / "logs" / f"iterative_local_{model}_{task}.log",
                    env=env, resume=bool(args.resume), gpu_ids=gpu_ids,
                ))

            for task in core_tasks:
                source_dir = OUTPUT_ROOT / "mainline" / model / task
                source_json = prediction_from(source_dir)
                output_dir = ABLATION_ROOT / "first_stage_only" / model / task
                command = [
                    str(args.python), str(FIRST_STAGE),
                    "--model", model, "--task", task,
                    "--source-json", str(source_json),
                    "--gpus", gpus, "--output-dir", str(output_dir),
                ]
                records.append(run_job(
                    name=f"first_stage_only_{model}_{task}", command=command, output_dir=output_dir,
                    log_path=PROVENANCE / "logs" / f"first_stage_only_{model}_{task}.log",
                    env=env, resume=bool(args.resume), gpu_ids=gpu_ids,
                ))

        write_runtime_table(records, suffix=suffix)
        post_commands: list[list[str]] = []
        if full_scope:
            post_commands = [
                [str(args.python), str(TABLE_BUILDER)],
                [str(args.python), str(LOCALIZATION)],
                [str(args.python), str(LOCK_METHOD)],
            ]
            for command in post_commands:
                subprocess.run(command, cwd=str(ROOT), env=env, check=True)

        status = {
            "status": "complete" if full_scope else "complete_selected_scope",
            "finished_at": datetime.now().astimezone().isoformat(),
            "frozen_config": str(FROZEN_CONFIG_PATH),
            "models": list(models),
            "tasks": list(tasks),
            "core_tasks": list(core_tasks),
            "records": records,
            "post_commands": post_commands,
            "finalization_deferred": not full_scope,
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[complete] final campaign status={status_path}")
    except Exception as exc:
        status = {
            "status": "failed",
            "failed_at": datetime.now().astimezone().isoformat(),
            "error": str(exc),
            "records": records,
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
