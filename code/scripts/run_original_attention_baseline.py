from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = PACKAGE_ROOT / "code" / "runtime"
CODE_ROOT = PACKAGE_ROOT / "code"


def pythonpath_env() -> str:
    parts = [str(RUNTIME), str(CODE_ROOT), str(PACKAGE_ROOT)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(CODE_ROOT))

from info import model_to_fullname, task_to_image_path, task_to_question_path  # noqa: E402
from blip_methods import (  # noqa: E402
    gradient_attention_blip,
    pure_gradient_blip,
    rel_attention_blip,
)
from llava_methods import (  # noqa: E402
    gradient_attention_llava,
    pure_gradient_llava,
    rel_attention_llava,
)
from qwen2_5_methods import (  # noqa: E402
    gradient_attention_qwen2_5,
    pure_gradient_qwen2_5,
    rel_attention_qwen2_5,
)
from top1_hotspot.adapters import load_model_adapter  # noqa: E402
from utils import bbox_from_att_image_adaptive, high_res  # noqa: E402


METHOD_DISPLAY_NAMES = {
    "rel_att": "Rel-Att",
    "rel_att_high": "Rel-Att-HR",
    "grad_att": "Grad-Att",
    "grad_att_high": "Grad-Att-HR",
    "pure_grad": "Pure-Grad",
    "pure_grad_high": "Pure-Grad-HR",
}
SUPPORTED_MODELS = ("llava", "blip", "qwen2_5")
SUPPORTED_TASKS = ("vstar", "hrbench_4k", "hrbench_8k")
SUPPORTED_METHODS = (
    "rel_att",
    "rel_att_high",
    "grad_att",
    "grad_att_high",
    "pure_grad",
    "pure_grad_high",
)


def dumps_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce original attention-crop baselines in the package."
    )
    parser.add_argument("--model", default="llava", choices=SUPPORTED_MODELS)
    parser.add_argument("--task", default="vstar", choices=SUPPORTED_TASKS)
    parser.add_argument("--method", default="rel_att_high", choices=SUPPORTED_METHODS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--question-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true", default=True)
    parser.add_argument("--no-evaluate", action="store_false", dest="evaluate")
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    cap_tag = f"{int(args.cap)}" if args.cap is not None else "full"
    return (
        PACKAGE_ROOT
        / "outputs"
        / "original_attention_baselines"
        / f"{args.task}_{args.model}_{args.method}_{cap_tag}"
    )


def question_path_for_task(task: str, override: str | None = None) -> Path:
    if override:
        return Path(override)
    return Path(task_to_question_path[task])


def image_path_for_row(
    task: str, row: dict[str, Any], image_root_override: str | None = None
) -> Path:
    image_root = Path(image_root_override) if image_root_override else Path(task_to_image_path[task])
    if "image_path" in row:
        image_path = Path(str(row["image_path"]))
        return image_path if image_path.is_absolute() else image_root / image_path
    if "image_id" in row:
        return image_root / f"{row['image_id']}.jpg"
    raise KeyError("row must contain image_path or image_id")


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("id")): row for row in rows if row.get("id") is not None}


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(rows), encoding="utf-8")


def parse_gpu_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    gpus = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def split_index_ranges(start_index: int, end_index: int, shard_count: int) -> list[tuple[int, int]]:
    total = int(end_index) - int(start_index)
    if total <= 0:
        raise RuntimeError(f"empty range: start={start_index}, end={end_index}")
    shard_count = max(1, min(int(shard_count), total))
    ranges: list[tuple[int, int]] = []
    for shard_index in range(shard_count):
        shard_start = int(start_index) + total * shard_index // shard_count
        shard_end = int(start_index) + total * (shard_index + 1) // shard_count
        if shard_start < shard_end:
            ranges.append((int(shard_start), int(shard_end)))
    return ranges


def evaluate_output(output_dir: Path, output_path: Path, model: str, task: str, method: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_env()
    subprocess.run(
        [
            sys.executable,
            str(RUNTIME / "get_score.py"),
            "--data_dir",
            str(output_dir),
            "--save_path",
            str(output_dir),
        ],
        cwd=str(PACKAGE_ROOT),
        env=env,
        check=True,
    )


def write_manifest(
    path: Path,
    *,
    status: str,
    args: argparse.Namespace,
    output_path: Path,
    question_path: Path,
    start_index: int,
    end_index: int,
    selected_count: int,
    completed_count: int,
    shards: list[dict[str, Any]] | None = None,
    last_sample_id: str | None = None,
    last_offset: int | None = None,
) -> None:
    high_res_enabled = str(args.method).endswith("_high")
    heatmap_name = str(args.method).removesuffix("_high")
    model_prefix = {
        "llava": "llava",
        "blip": "instructblip",
        "qwen2_5": "qwen2_5",
    }[str(args.model)]
    crop_base_size = {
        "llava": 336,
        "blip": 224,
        "qwen2_5": 336,
    }[str(args.model)]
    manifest = {
        "status": status,
        "model": str(args.model),
        "task": str(args.task),
        "method": str(args.method),
        "method_display_name": METHOD_DISPLAY_NAMES[str(args.method)],
        "question_path": str(question_path),
        "range": [int(start_index), int(end_index)],
        "cap": int(args.cap) if args.cap is not None else None,
        "count": int(selected_count),
        "completed_count": int(completed_count),
        "last_sample_id": last_sample_id,
        "last_offset": last_offset,
        "output": str(output_path),
        "evaluation_report": str(path.parent / "evaluation_report.json"),
        "evaluation_csv": str(path.parent / "evaluation_report.csv"),
        "gpu": args.gpu,
        "gpus": parse_gpu_list(args.gpus),
        "shards": shards or [],
        "bbox_policy": "original_bbox_from_att_image_adaptive",
        "crop_base_size": int(crop_base_size),
        "heatmap_mode": (
            f"{model_prefix}_{heatmap_name}_highres_tile1024"
            if high_res_enabled
            else f"{model_prefix}_{heatmap_name}"
        ),
        "high_res_tile_threshold": 1024 if high_res_enabled else None,
        "high_res_policy": (
            "split_non_overlapping_tiles_then_stitch_heatmaps"
            if high_res_enabled
            else None
        ),
        "final_input_policy": "original_plus_crop",
    }
    path.write_text(dumps_json(manifest), encoding="utf-8")


def llava_single_answer(image: Image.Image, prompt: str, model: Any, processor: Any, max_new_tokens: int) -> str:
    import torch

    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(
        model.device, torch.bfloat16
    )
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if "ASSISTANT: " in text:
        text = text.split("ASSISTANT: ", 1)[1]
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text.strip()


def llava_original_plus_crop_answer(
    image: Image.Image,
    crop_image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = processor(images=[image, crop_image], text=prompt, return_tensors="pt", padding=True).to(
        model.device, torch.bfloat16
    )
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    if "ASSISTANT: " in text:
        text = text.split("ASSISTANT: ", 1)[1]
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text.strip()


def load_llava(device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    local_files_only = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_to_fullname["llava"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=local_files_only,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_to_fullname["llava"],
        local_files_only=local_files_only,
    )
    model.eval()
    return model, processor


def run_sample(
    *,
    row: dict[str, Any],
    task: str,
    adapter: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image = Image.open(image_path_for_row(task, row, args.image_root)).convert("RGB")
    short_question = str(row.get("short_question") or row["question"])
    question = str(row["question"])
    attention_prompt = adapter.attention_prompt(short_question)
    answer_prompt = adapter.answer_prompt(question)
    general_prompt = adapter.general_prompt

    original_answer = adapter.original_answer_fn(
        image, answer_prompt, adapter.model, adapter.processor, int(args.max_new_tokens)
    )
    method_to_func = {
        ("llava", "rel_att"): rel_attention_llava,
        ("llava", "rel_att_high"): rel_attention_llava,
        ("llava", "grad_att"): gradient_attention_llava,
        ("llava", "grad_att_high"): gradient_attention_llava,
        ("llava", "pure_grad"): pure_gradient_llava,
        ("llava", "pure_grad_high"): pure_gradient_llava,
        ("blip", "rel_att"): rel_attention_blip,
        ("blip", "rel_att_high"): rel_attention_blip,
        ("blip", "grad_att"): gradient_attention_blip,
        ("blip", "grad_att_high"): gradient_attention_blip,
        ("blip", "pure_grad"): pure_gradient_blip,
        ("blip", "pure_grad_high"): pure_gradient_blip,
        ("qwen2_5", "rel_att"): rel_attention_qwen2_5,
        ("qwen2_5", "rel_att_high"): rel_attention_qwen2_5,
        ("qwen2_5", "grad_att"): gradient_attention_qwen2_5,
        ("qwen2_5", "grad_att_high"): gradient_attention_qwen2_5,
        ("qwen2_5", "pure_grad"): pure_gradient_qwen2_5,
        ("qwen2_5", "pure_grad_high"): pure_gradient_qwen2_5,
    }
    map_func = method_to_func[(str(args.model), str(args.method))]
    if str(args.method).endswith("_high"):
        heatmap = high_res(
            map_func,
            image,
            attention_prompt,
            general_prompt,
            adapter.model,
            adapter.processor,
        )
    else:
        heatmap = map_func(
            image,
            attention_prompt,
            general_prompt,
            adapter.model,
            adapter.processor,
        )
    bbox = tuple(
        int(v)
        for v in bbox_from_att_image_adaptive(
            heatmap,
            image.size,
            int(adapter.vit_input_size),
        )
    )
    crop_image = image.crop(bbox)
    crop_answer = adapter.crop_answer_fn(
        image,
        crop_image,
        answer_prompt,
        adapter.model,
        adapter.processor,
        int(args.max_new_tokens),
    )

    out_row = dict(row)
    out_row.update(
        {
            "original_answer": original_answer,
            "crop_answer": crop_answer,
            "bbox": list(bbox),
            "crop_method": str(args.method),
            "method_display_name": METHOD_DISPLAY_NAMES[str(args.method)],
            "qpes_heatmap_mode": (
                f"{str(args.model)}_{str(args.method).removesuffix('_high')}_highres_tile1024"
                if str(args.method).endswith("_high")
                else f"{str(args.model)}_{str(args.method)}"
            ),
            "qpes_high_res_tile_threshold": 1024 if str(args.method).endswith("_high") else None,
            "qpes_crop_base_size": int(adapter.vit_input_size),
            "qpes_initial_heatmap_shape": [int(v) for v in heatmap.shape],
        }
    )
    return out_row


def build_shard_command(
    *,
    args: argparse.Namespace,
    gpu: str,
    shard_start: int,
    shard_end: int,
    shard_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--method",
        str(args.method),
        "--start-index",
        str(shard_start),
        "--end-index",
        str(shard_end),
        "--output-dir",
        str(shard_dir),
        "--gpu",
        str(gpu),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--log-every",
        str(args.log_every),
        "--no-evaluate",
    ]
    if args.cap is not None:
        cmd.extend(["--cap", str(args.cap)])
    if args.question_path:
        cmd.extend(["--question-path", str(args.question_path)])
    if args.image_root:
        cmd.extend(["--image-root", str(args.image_root)])
    if args.resume:
        cmd.append("--resume")
    return cmd


def run_multigpu(args: argparse.Namespace) -> None:
    gpus = parse_gpu_list(args.gpus)
    question_path = question_path_for_task(str(args.task), args.question_path)
    rows = json.loads(question_path.read_text(encoding="utf-8"))
    cap_end = len(rows) if args.cap is None else min(int(args.cap), len(rows))
    end_index = cap_end if args.end_index is None else min(int(args.end_index), cap_end)
    start_index = int(args.start_index)
    selected_rows = rows[start_index:int(end_index)]
    if not selected_rows:
        raise RuntimeError(f"no selected rows for task={args.task}")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}-{args.task}-{args.method}.json"
    shard_root = output_dir / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_infos: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = pythonpath_env()
    env_base.setdefault("HF_HUB_OFFLINE", "1")
    env_base.setdefault("TRANSFORMERS_OFFLINE", "1")

    running: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    launch_delay_sec = float(os.environ.get("ORIGINAL_BASELINE_SHARD_LAUNCH_DELAY_SEC", "0") or 0)
    for index, (gpu, (shard_start, shard_end)) in enumerate(
        zip(gpus, split_index_ranges(start_index, int(end_index), len(gpus)))
    ):
        shard_dir = shard_root / f"shard_{shard_start:03d}_{shard_end:03d}_gpu{gpu}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "run.log"
        cmd = build_shard_command(
            args=args,
            gpu=gpu,
            shard_start=shard_start,
            shard_end=shard_end,
            shard_dir=shard_dir,
        )
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        info = {
            "index": int(index),
            "gpu": str(gpu),
            "range": [int(shard_start), int(shard_end)],
            "output": str(shard_dir / f"{args.model}-{args.task}-{args.method}.json"),
            "log": str(log_path),
            "return_code": None,
            "completed_count": 0,
        }
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PACKAGE_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        running.append((proc, log_file, info))
        if launch_delay_sec > 0 and index < len(gpus) - 1:
            time.sleep(launch_delay_sec)

    for proc, log_file, info in running:
        return_code = proc.wait()
        log_file.close()
        info["return_code"] = int(return_code)
        shard_output = Path(info["output"])
        if shard_output.exists():
            info["completed_count"] = len(json.loads(shard_output.read_text(encoding="utf-8")))
        shard_infos.append(info)
        if return_code != 0:
            failures.append(info)

    manifest_path = output_dir / "run_manifest.json"
    if failures:
        write_manifest(
            manifest_path,
            status="failed",
            args=args,
            output_path=output_path,
            question_path=question_path,
            start_index=start_index,
            end_index=int(end_index),
            selected_count=len(selected_rows),
            completed_count=sum(int(info["completed_count"]) for info in shard_infos),
            shards=shard_infos,
        )
        raise RuntimeError(f"{len(failures)} shard(s) failed; see {shard_root}")

    by_id: dict[str, dict[str, Any]] = {}
    for info in shard_infos:
        shard_rows = json.loads(Path(info["output"]).read_text(encoding="utf-8"))
        for row in shard_rows:
            sample_id = str(row["id"])
            if sample_id in by_id:
                raise RuntimeError(f"duplicate sample id across shards: {sample_id}")
            by_id[sample_id] = row
    final_rows = [by_id[str(row["id"])] for row in selected_rows if str(row["id"]) in by_id]
    if len(final_rows) != len(selected_rows):
        raise RuntimeError(f"merged output incomplete: got {len(final_rows)}, expected {len(selected_rows)}")
    save_rows(output_path, final_rows)
    if args.evaluate:
        evaluate_output(output_dir, output_path, str(args.model), str(args.task), str(args.method))
    write_manifest(
        manifest_path,
        status="complete",
        args=args,
        output_path=output_path,
        question_path=question_path,
        start_index=start_index,
        end_index=int(end_index),
        selected_count=len(selected_rows),
        completed_count=len(final_rows),
        shards=shard_infos,
    )


def main() -> None:
    args = parse_args()
    if args.gpus:
        run_multigpu(args)
        return

    if args.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["PYTHONPATH"] = pythonpath_env()

    import torch

    question_path = question_path_for_task(str(args.task), args.question_path)
    rows = json.loads(question_path.read_text(encoding="utf-8"))
    cap_end = len(rows) if args.cap is None else min(int(args.cap), len(rows))
    end_index = cap_end if args.end_index is None else min(int(args.end_index), cap_end)
    selected_rows = rows[int(args.start_index):int(end_index)]
    if not selected_rows:
        raise RuntimeError(f"no selected rows for task={args.task}")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}-{args.task}-{args.method}.json"
    manifest_path = output_dir / "run_manifest.json"
    existing = load_existing_rows(output_path) if args.resume else {}
    outputs = [existing[str(row["id"])] for row in selected_rows if str(row.get("id")) in existing]
    processed = {str(row.get("id")) for row in outputs}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = load_model_adapter(str(args.model), device)

    for offset, row in enumerate(selected_rows, start=int(args.start_index)):
        sample_id = str(row["id"])
        if sample_id in processed:
            print(f"[skip] task={args.task} model={args.model} id={sample_id}", flush=True)
            continue
        out_row = run_sample(
            row=row,
            task=str(args.task),
            adapter=adapter,
            args=args,
        )
        outputs.append(out_row)
        processed.add(sample_id)
        save_rows(output_path, outputs)
        write_manifest(
            manifest_path,
            status="running",
            args=args,
            output_path=output_path,
            question_path=question_path,
            start_index=int(args.start_index),
            end_index=int(end_index),
            selected_count=len(selected_rows),
            completed_count=len(outputs),
            last_sample_id=sample_id,
            last_offset=int(offset),
        )
        should_log = (
            int(args.log_every) <= 1
            or len(outputs) == 1
            or len(outputs) == len(selected_rows)
            or len(outputs) % int(args.log_every) == 0
        )
        if should_log:
            print(
                f"[done] task={args.task} model={args.model} method={args.method} "
                f"id={sample_id} count={len(outputs)}/{len(selected_rows)}",
                flush=True,
            )

    ordered = {str(row["id"]): row for row in outputs}
    final_rows = [ordered[str(row["id"])] for row in selected_rows if str(row["id"]) in ordered]
    if len(final_rows) != len(selected_rows):
        raise RuntimeError(f"output incomplete: got {len(final_rows)}, expected {len(selected_rows)}")
    save_rows(output_path, final_rows)
    if args.evaluate:
        evaluate_output(output_dir, output_path, str(args.model), str(args.task), str(args.method))
    write_manifest(
        manifest_path,
        status="complete",
        args=args,
        output_path=output_path,
        question_path=question_path,
        start_index=int(args.start_index),
        end_index=int(end_index),
        selected_count=len(selected_rows),
        completed_count=len(final_rows),
    )


if __name__ == "__main__":
    main()
