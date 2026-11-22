#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import shutil
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNTIME = REPO / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def pythonpath_env() -> str:
    parts = [str(REPO), str(RUNTIME)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)

from top1_hotspot.adapters import load_model_adapter  # noqa: E402
from top1_hotspot.components import hotspot_component_square_bbox  # noqa: E402
from top1_hotspot.adaptive_stop import component_short_ratio  # noqa: E402
from top1_hotspot.config import (  # noqa: E402
    ADAPTIVE_STOP_RATIO,
    COMMON_MIN_CROP_SIDE,
    FROZEN_CONFIG,
    METHOD_CN_NAME,
    METHOD_DISPLAY_NAME,
    METHOD_FULL_NAME,
    SUPPORTED_MODELS,
    frozen_config_sha256,
    resolve_model_params,
)
from top1_hotspot.geometry import (  # noqa: E402
    HALVING_INITIAL_AREA_RATIO,
    HALVING_RATIO_DECAY,
    bbox_to_int_tuple,
    build_stage_targets,
    dumps_json,
    jsonable,
    sanitize_heatmap,
)
from top1_hotspot.high_res import tile_heatmap_mosaic  # noqa: E402
from top1_hotspot.heatmap_stats import compute_normalized_heatmap_entropy  # noqa: E402
from top1_hotspot.pipeline import (  # noqa: E402
    LOCALIZATION_STRATEGIES,
    compact_stage,
    run_localization_chain,
)
from scoring_contract import SCORER_CONTRACT_VERSION  # noqa: E402

HIGH_RES_TILE_THRESHOLD = int(FROZEN_CONFIG["shared"]["high_res_tile_threshold"])
CALIBRATION_TOPMASS_ENV = "PHF_CALIBRATION_TOPMASS"
SENSITIVITY_TOPMASS_ENV = "PHF_SENSITIVITY_TOPMASS"
SENSITIVITY_REASON_ENV = "PHF_SENSITIVITY_REASON"
SENSITIVITY_VARIANTS = {
    "min224": {"min_crop_side": 224},
    "min448": {"min_crop_side": 448},
    "initial025": {"initial_area_ratio": 0.25},
    "initial075": {"initial_area_ratio": 0.75},
    "decay025": {"area_ratio_decay": 0.25},
    "decay075": {"area_ratio_decay": 0.75},
    "rho045": {"topmass": 0.45},
    "rho050": {"topmass": 0.50},
    "rho060": {"topmass": 0.60},
    "score_mass": {"component_score": "mass"},
    "score_no_area": {"component_score": "peak_mean"},
}
MANIFEST_SCHEMA_VERSION = 2
_CODE_PROVENANCE_CACHE: dict[str, Any] | None = None
_RUNTIME_PROVENANCE_CACHE: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_provenance() -> dict[str, Any]:
    """Return a cheap, reproducible code-state fingerprint for manifests."""
    global _CODE_PROVENANCE_CACHE
    if _CODE_PROVENANCE_CACHE is not None:
        return dict(_CODE_PROVENANCE_CACHE)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PACKAGE_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=str(PACKAGE_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        _CODE_PROVENANCE_CACHE = {
            "revision": revision,
            "working_tree_dirty": bool(status),
            "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        _CODE_PROVENANCE_CACHE = {
            "revision": None,
            "working_tree_dirty": None,
            "working_tree_status_sha256": None,
        }
    return dict(_CODE_PROVENANCE_CACHE)


def runtime_provenance() -> dict[str, Any]:
    global _RUNTIME_PROVENANCE_CACHE
    if _RUNTIME_PROVENANCE_CACHE is not None:
        return dict(_RUNTIME_PROVENANCE_CACHE)
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "Pillow", "pandas", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    model_references: dict[str, str] = {}
    try:
        from info import model_to_fullname

        model_references = {str(key): str(value) for key, value in model_to_fullname.items()}
    except (ImportError, AttributeError):
        pass
    _RUNTIME_PROVENANCE_CACHE = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "model_references": model_references,
    }
    return dict(_RUNTIME_PROVENANCE_CACHE)


def manifest_identity(
    args: argparse.Namespace,
    question_path: Path,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "task": str(args.task),
        "method": str(args.method_name),
        "question_path": str(question_path.resolve()),
        "question_sha256": sha256_file(question_path),
        "range": [int(start_index), int(end_index)],
        "cap": int(args.cap) if args.cap is not None else None,
        "topmass": float(args.topmass),
        "sensitivity_variant": getattr(args, "sensitivity_variant", None),
        "localization_strategy": str(args.localization_strategy),
        "attention_question_source": str(args.attention_question_source),
        "min_crop_side": int(args.min_crop_side),
        "initial_heatmap_bootstrap": str(args.initial_heatmap_bootstrap),
        "local_heatmap_bootstrap": str(args.local_heatmap_bootstrap),
        "adaptive_stop_hotspot_ratio": getattr(args, "adaptive_stop_hotspot_ratio", None),
        "frozen_config_sha256": frozen_config_sha256(),
        "calibration_manifest_sha256": calibration_manifest_sha256(),
        "scorer_contract_version": SCORER_CONTRACT_VERSION,
        "code": code_provenance(),
        "runtime": runtime_provenance(),
    }


def validate_resume_manifest(
    path: Path,
    expected_identity: dict[str, Any],
) -> str:
    """Validate new manifests while keeping legacy outputs readable."""
    if not path.is_file():
        return "missing_manifest"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("manifest_schema_version", 0)) < MANIFEST_SCHEMA_VERSION:
        return "legacy_manifest"
    actual = manifest.get("run_identity")
    if actual != expected_identity:
        mismatches = sorted(
            key for key in set(expected_identity) | set(actual or {})
            if (actual or {}).get(key) != expected_identity.get(key)
        )
        raise RuntimeError(
            f"resume identity mismatch in {path}; differing fields: {mismatches}"
        )
    return "validated_manifest"


def ensure_output_contract(
    output_path: Path,
    manifest_path: Path,
    *,
    resume: bool,
    allow_existing: bool,
) -> None:
    if resume or allow_existing:
        return
    conflicts = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if conflicts:
        raise RuntimeError(
            "refusing to overwrite existing output; use --resume after identity "
            f"validation or --allow-existing-output: {conflicts}"
        )


def calibration_manifest_path() -> Path:
    path = Path(str(FROZEN_CONFIG["calibration"]["manifest"]))
    return path if path.is_absolute() else PACKAGE_ROOT / path


def calibration_manifest_sha256() -> str | None:
    path = calibration_manifest_path()
    if path.is_file():
        return sha256_file(path)
    if FROZEN_CONFIG["status"] == "locked" and not path.exists():
        # The release uses fixed parameters without calibration artifacts.
        return None
    raise FileNotFoundError(f"missing calibration manifest: {path}")


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    calibration_override = os.environ.get(CALIBRATION_TOPMASS_ENV)
    sensitivity_override = os.environ.get(SENSITIVITY_TOPMASS_ENV)
    variant_name = getattr(args, "sensitivity_variant", None)
    variant = SENSITIVITY_VARIANTS[variant_name] if variant_name else {}
    if variant_name and (
        calibration_override is not None
        or sensitivity_override is not None
        or args.model != "llava"
        or args.localization_strategy != "phf"
    ):
        raise ValueError("sensitivity variants require LLaVA PHF without topmass environment overrides")
    args.initial_area_ratio = variant.get("initial_area_ratio", HALVING_INITIAL_AREA_RATIO)
    args.area_ratio_decay = variant.get("area_ratio_decay", HALVING_RATIO_DECAY)
    args.component_score = variant.get("component_score", "default")
    args.stage_policy = (
        "halve_area_ratio_until_min_crop_side"
        if args.area_ratio_decay == HALVING_RATIO_DECAY
        else "area_ratio_decay_until_min_crop_side"
    )
    if calibration_override is not None and sensitivity_override is not None:
        raise RuntimeError("calibration and sensitivity topmass overrides are mutually exclusive")
    if calibration_override is not None and FROZEN_CONFIG.get("status") != "calibration_pending":
        raise RuntimeError(
            f"{CALIBRATION_TOPMASS_ENV} is only valid while frozen config status is calibration_pending"
        )
    resolved = resolve_model_params(
        model=str(args.model),
        topmass=(
            float(calibration_override)
            if calibration_override is not None
            else float(sensitivity_override)
            if sensitivity_override is not None
            else variant.get("topmass")
        ),
        min_crop_side=variant.get("min_crop_side", COMMON_MIN_CROP_SIDE),
    )
    args.topmass = float(resolved["topmass"])
    args.min_crop_side = int(resolved["min_crop_side"])
    args.blip_bg_sub_alpha = float(resolved["blip_bg_sub_alpha"])
    args.method_name = str(resolved["method_name"])
    args.initial_heatmap_bootstrap = str(FROZEN_CONFIG["shared"]["initial_heatmap_bootstrap"])
    args.adaptive_stop_hotspot_ratio = ADAPTIVE_STOP_RATIO
    args.sensitivity_reason = (
        os.environ.get(SENSITIVITY_REASON_ENV)
        if sensitivity_override is not None
        else None
    )
    if sensitivity_override is not None and not args.sensitivity_reason:
        raise RuntimeError(
            f"{SENSITIVITY_REASON_ENV} is required with {SENSITIVITY_TOPMASS_ENV}"
        )
    args.method_name = f"{args.method_name}_init_{args.initial_heatmap_bootstrap}_stop1"
    if str(args.local_heatmap_bootstrap) != "full":
        args.method_name = f"{args.method_name}_local_{str(args.local_heatmap_bootstrap)}"
    if str(args.localization_strategy) != "phf":
        args.method_name = f"{args.method_name}_{str(args.localization_strategy)}"
    if variant_name:
        args.method_name = f"{args.method_name}_{variant_name}"
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PHF-Crop (Progressive Hotspot Fusion Crop): triggered "
            "tile1024 initial heatmap, adaptive stop, and halved area-ratio "
            "stages down to the frozen minimum crop side."
        )
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--task", default="vstar")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--question-path", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", default="")
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated GPU ids for one-command sharded execution. "
            "Each shard writes to output-dir/_shards and the parent merges "
            "then evaluates after all shards finish."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--attention-question-source",
        choices=("short", "full"),
        default="short",
        help="Use short_question when present, or always use the full question.",
    )
    parser.add_argument(
        "--localization-strategy",
        choices=LOCALIZATION_STRATEGIES,
        default="phf",
        help=(
            "Localization chain. 'phf' maps every local heatmap back to the original "
            "canvas and fuses history; 'iterative_local' keeps the same stage schedule "
            "but selects each next crop only from the latest local heatmap."
        ),
    )
    parser.add_argument(
        "--local-heatmap-bootstrap",
        choices=("full", "tile1024"),
        default="full",
        help="Heatmap policy for non-terminal PHF crop stages; tile1024 tiles only crops whose longest side exceeds 1024.",
    )
    parser.add_argument(
        "--sensitivity-variant",
        choices=tuple(SENSITIVITY_VARIANTS),
        help="Change one parameter in the LLaVA sensitivity study.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing output path without resume validation.",
    )
    parser.add_argument(
        "--legacy-default-output-dir",
        action="store_true",
        help="Use the pre-schema task/model/cap output directory naming.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep successful multi-GPU _shards after merge. Default: delete them.",
    )
    parser.add_argument("--evaluate", action="store_true", default=True)
    parser.add_argument("--no-evaluate", action="store_false", dest="evaluate")
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    model_tag = str(args.model).replace("qwen2_5", "qwen")
    task_tag = str(args.task)
    cap_tag = f"{int(args.cap)}" if args.cap is not None else "full"
    if bool(args.legacy_default_output_dir):
        return PACKAGE_ROOT / "outputs" / f"{task_tag}_{model_tag}_phf_crop_final_{cap_tag}"
    method_tag = str(args.method_name).replace("/", "_")
    return PACKAGE_ROOT / "outputs" / f"{task_tag}_{model_tag}_{method_tag}_{cap_tag}"


def question_path_for_task(task: str, override: str | None = None) -> Path:
    if override:
        return Path(override)

    from info import task_to_question_path

    if task not in task_to_question_path:
        raise ValueError(f"unsupported task: {task}")
    return Path(task_to_question_path[task])


def image_path_for_row(task: str, row: dict[str, Any], image_root_override: str | None = None) -> Path:
    from info import task_to_image_path

    image_root = Path(image_root_override) if image_root_override else Path(task_to_image_path[task])
    if "image_path" in row:
        image_path = Path(str(row["image_path"]))
        return image_path if image_path.is_absolute() else image_root / image_path
    if "image_id" in row:
        return image_root / f"{row['image_id']}.jpg"
    raise KeyError("row must contain image_path or image_id")


def attention_question(row: dict[str, Any], source: str) -> str:
    if source == "short":
        return str(row.get("short_question") or row["question"])
    if source == "full":
        return str(row["question"])
    raise ValueError(f"unsupported attention question source: {source}")


def load_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("id")): row for row in rows if row.get("id") is not None}


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(rows), encoding="utf-8")


def evaluate_output(output_dir: Path, python_exe: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_env()
    subprocess.run(
        [
            python_exe,
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


def append_optional_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


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
        "--attention-question-source",
        str(args.attention_question_source),
        "--localization-strategy",
        str(args.localization_strategy),
        "--local-heatmap-bootstrap",
        str(args.local_heatmap_bootstrap),
        "--log-every",
        str(args.log_every),
        "--no-evaluate",
    ]
    append_optional_arg(cmd, "--cap", args.cap)
    append_optional_arg(cmd, "--question-path", args.question_path)
    append_optional_arg(cmd, "--image-root", args.image_root)
    append_optional_arg(cmd, "--sensitivity-variant", getattr(args, "sensitivity_variant", None))
    if args.resume:
        cmd.append("--resume")
    if args.allow_existing_output:
        cmd.append("--allow-existing-output")
    return cmd


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-int(max_lines) :])


def write_multigpu_manifest(
    path: Path,
    *,
    status: str,
    args: argparse.Namespace,
    output_path: Path,
    question_path: Path,
    start_index: int,
    end_index: int,
    completed_count: int,
    shard_infos: list[dict[str, Any]],
    resume_source: str | None = None,
) -> None:
    run_identity = manifest_identity(args, question_path, start_index, end_index)
    stop_threshold = getattr(args, "adaptive_stop_hotspot_ratio", None)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "execution_mode": "multi_gpu",
        "status": status,
        "resume": bool(args.resume),
        "resume_source": resume_source,
        "run_identity": run_identity,
        "runtime_provenance": runtime_provenance(),
        "model": str(args.model),
        "task": str(args.task),
        "method": str(args.method_name),
        "method_display_name": METHOD_DISPLAY_NAME,
        "method_chinese_name": METHOD_CN_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "question_path": str(question_path),
        "range": [int(start_index), int(end_index)],
        "cap": int(args.cap) if args.cap is not None else None,
        "count": int(end_index) - int(start_index),
        "completed_count": int(completed_count),
        "output": str(output_path),
        "evaluation_report": str(path.parent / "evaluation_report.json"),
        "evaluation_csv": str(path.parent / "evaluation_report.csv"),
        "scorer_contract": str(path.parent / "scorer_contract.json"),
        "scorer_contract_version": SCORER_CONTRACT_VERSION,
        "gpus": parse_gpu_list(args.gpus),
        "shards": shard_infos,
        "shards_retained": bool(args.keep_shards or status != "complete"),
        "stage_policy": args.stage_policy,
        "initial_area_ratio": args.initial_area_ratio,
        "ratio_decay": args.area_ratio_decay,
        "component_score": args.component_score,
        "normalize_contributions": "sum",
        "fusion": (
            "coverage_average"
            if str(args.localization_strategy) == "phf"
            else "none_last_local_only"
        ),
        "localization_strategy": str(args.localization_strategy),
        "bootstrap": str(args.initial_heatmap_bootstrap),
        "initial_heatmap_bootstrap": str(args.initial_heatmap_bootstrap),
        "local_heatmap_bootstrap": str(args.local_heatmap_bootstrap),
        "high_res_tile_threshold": (
            HIGH_RES_TILE_THRESHOLD
            if str(args.initial_heatmap_bootstrap) == "tile1024"
            else None
        ),
        "selection_policy": "top1_hotspot_component_center",
        "topmass": float(args.topmass),
        "topmass_mask_policy": "threshold_ge_with_ties; actual covered fraction recorded per stage",
        "attention_question_source": str(args.attention_question_source),
        "min_crop_side": int(args.min_crop_side),
        "frozen_config_sha256": frozen_config_sha256(),
        "calibration_manifest_sha256": calibration_manifest_sha256(),
        "sensitivity_override": (
            {"topmass": float(args.topmass), "reason": str(args.sensitivity_reason)}
            if args.sensitivity_reason
            else None
        ),
        "qwen_visual_tokens": (
            int(FROZEN_CONFIG["models"]["qwen2_5"]["visual_tokens"])
            if str(args.model) == "qwen2_5"
            else None
        ),
        "qwen_max_pixels": (
            int(FROZEN_CONFIG["models"]["qwen2_5"]["max_pixels"])
            if str(args.model) == "qwen2_5"
            else None
        ),
        "blip_bg_sub_alpha": float(args.blip_bg_sub_alpha),
        "final_input_policy": final_input_policy(
            str(args.model),
            adaptive_stop_enabled(args),
        ),
        "adaptive_stop": {
            "hotspot_ratio": stop_threshold,
            "operator": ">=",
            "ratio_side": "short",
            "valid_component_required": True,
            "stop_reasons": ["hotspot_fills_candidate", "min_crop_side"],
            "final_bbox_semantics": "current_centered_square_crop",
            "execution": (
                "early_terminate_before_selected_stage_local_heatmap"
                if stop_threshold is not None
                else "stop_disabled_full_schedule"
            ),
        },
    }
    path.write_text(dumps_json(manifest), encoding="utf-8")


def run_multigpu(args: argparse.Namespace) -> None:
    if args.gpu:
        raise ValueError("use either --gpu for single-process mode or --gpus for sharded mode, not both")

    gpus = parse_gpu_list(args.gpus)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["PYTHONPATH"] = pythonpath_env()
    os.environ["QPES_BLIP_BG_SUB_ALPHA"] = str(args.blip_bg_sub_alpha)

    question_path = question_path_for_task(str(args.task), args.question_path)
    all_rows = json.loads(question_path.read_text(encoding="utf-8"))
    cap_end = len(all_rows) if args.cap is None else min(int(args.cap), len(all_rows))
    end_index = cap_end if args.end_index is None else min(int(args.end_index), cap_end)
    start_index = int(args.start_index)
    selected_rows = all_rows[start_index:int(end_index)]
    if not selected_rows:
        raise RuntimeError(f"no selected rows for task={args.task}")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_path = output_dir / f"{args.model}-{args.task}-{args.method_name}.json"
    manifest_path = output_dir / "run_manifest.json"
    expected_identity = manifest_identity(args, question_path, start_index, int(end_index))
    resume_source = (
        validate_resume_manifest(manifest_path, expected_identity)
        if args.resume
        else None
    )
    ensure_output_contract(
        output_path,
        manifest_path,
        resume=bool(args.resume),
        allow_existing=bool(args.allow_existing_output),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = output_dir / "_shards"
    if shard_root.exists() and not args.resume and not args.allow_existing_output:
        raise RuntimeError(
            f"refusing to reuse existing shard directory without --resume: {shard_root}"
        )
    shard_root.mkdir(parents=True, exist_ok=True)

    write_multigpu_manifest(
        manifest_path,
        status="running",
        args=args,
        output_path=output_path,
        question_path=question_path,
        start_index=start_index,
        end_index=int(end_index),
        completed_count=0,
        shard_infos=[],
        resume_source=resume_source,
    )

    ranges = split_index_ranges(start_index, int(end_index), len(gpus))
    active_gpus = gpus[: len(ranges)]
    shard_infos: list[dict[str, Any]] = []
    processes: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []

    for shard_index, ((shard_start, shard_end), gpu) in enumerate(zip(ranges, active_gpus)):
        shard_dir = shard_root / f"shard_{shard_start:06d}_{shard_end:06d}_gpu{gpu}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "run.log"
        shard_output = shard_dir / f"{args.model}-{args.task}-{args.method_name}.json"
        cmd = build_shard_command(
            args=args,
            gpu=gpu,
            shard_start=shard_start,
            shard_end=shard_end,
            shard_dir=shard_dir,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = pythonpath_env()
        log_file = log_path.open("w", encoding="utf-8")
        print(
            f"[shard-start] index={shard_index} gpu={gpu} range={shard_start}:{shard_end} "
            f"log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(PACKAGE_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        info = {
            "index": int(shard_index),
            "gpu": str(gpu),
            "range": [int(shard_start), int(shard_end)],
            "dir": str(shard_dir),
            "log": str(log_path),
            "output": str(shard_output),
        }
        shard_infos.append(info)
        processes.append((info, proc, log_file))

    failed: list[dict[str, Any]] = []
    for info, proc, log_file in processes:
        return_code = proc.wait()
        log_file.close()
        info["return_code"] = int(return_code)
        if return_code == 0:
            rows = load_existing_rows(Path(info["output"]))
            info["completed_count"] = len(rows)
            print(
                f"[shard-complete] index={info['index']} gpu={info['gpu']} "
                f"range={info['range']} count={len(rows)}",
                flush=True,
            )
        else:
            failed.append(info)
            print(
                f"[shard-failed] index={info['index']} gpu={info['gpu']} "
                f"range={info['range']} return_code={return_code}\n"
                f"{tail_text(Path(info['log']))}",
                flush=True,
            )

    write_multigpu_manifest(
        manifest_path,
        status="running",
        args=args,
        output_path=output_path,
        question_path=question_path,
        start_index=start_index,
        end_index=int(end_index),
        completed_count=sum(int(info.get("completed_count", 0)) for info in shard_infos),
        shard_infos=shard_infos,
        resume_source=resume_source,
    )
    if failed:
        write_multigpu_manifest(
            manifest_path,
            status="failed",
            args=args,
            output_path=output_path,
            question_path=question_path,
            start_index=start_index,
            end_index=int(end_index),
            completed_count=sum(int(info.get("completed_count", 0)) for info in shard_infos),
            shard_infos=shard_infos,
            resume_source=resume_source,
        )
        raise RuntimeError(f"{len(failed)} shard(s) failed; see logs under {shard_root}")

    merged_by_id: dict[str, dict[str, Any]] = {}
    for info in shard_infos:
        rows = json.loads(Path(info["output"]).read_text(encoding="utf-8"))
        for row in rows:
            sample_id = str(row.get("id"))
            if sample_id in merged_by_id:
                raise RuntimeError(f"duplicate sample id across shards: {sample_id}")
            merged_by_id[sample_id] = row

    final_rows = [
        merged_by_id[str(row["id"])]
        for row in selected_rows
        if str(row["id"]) in merged_by_id
    ]
    if len(final_rows) != len(selected_rows):
        missing = [
            str(row["id"])
            for row in selected_rows
            if str(row["id"]) not in merged_by_id
        ]
        raise RuntimeError(
            f"merged output incomplete: got {len(final_rows)}, "
            f"expected {len(selected_rows)}, missing={missing[:10]}"
        )
    save_rows(output_path, final_rows)
    print(f"[merged] output={output_path} count={len(final_rows)}", flush=True)

    if args.evaluate:
        evaluate_output(output_dir, sys.executable)

    write_multigpu_manifest(
        manifest_path,
        status="complete",
        args=args,
        output_path=output_path,
        question_path=question_path,
        start_index=start_index,
        end_index=int(end_index),
        completed_count=len(final_rows),
        shard_infos=shard_infos,
        resume_source=resume_source,
    )
    if not args.keep_shards:
        shutil.rmtree(shard_root)
        print(f"[shards-removed] path={shard_root}", flush=True)


def final_input_policy(model_name: str, adaptive_stop_enabled: bool = False) -> str:
    if model_name == "blip":
        return (
            "crop_only_adaptive_stop_crop"
            if adaptive_stop_enabled
            else "crop_only_final_crop"
        )
    if adaptive_stop_enabled:
        return "original_plus_adaptive_stop_crop"
    return "original_plus_final_crop"


def adaptive_stop_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "adaptive_stop_hotspot_ratio", None) is not None


def compute_initial_heatmap(
    *,
    image: Image.Image,
    adapter: Any,
    attention_prompt: str,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any] | None]:
    if str(args.initial_heatmap_bootstrap) == "full":
        return (
            adapter.heatmap_fn(
                image,
                attention_prompt,
                adapter.general_prompt,
                adapter.model,
                adapter.processor,
            ),
            None,
        )
    if str(args.initial_heatmap_bootstrap) == "tile1024":
        image_width, image_height = image.size
        if max(int(image_width), int(image_height)) <= HIGH_RES_TILE_THRESHOLD:
            return (
                adapter.heatmap_fn(
                    image,
                    attention_prompt,
                    adapter.general_prompt,
                    adapter.model,
                    adapter.processor,
                ),
                {
                    "mode": "full_below_tile_threshold",
                    "tile_size": HIGH_RES_TILE_THRESHOLD,
                    "tile_count": 1,
                    "image_size": [int(image_width), int(image_height)],
                    "triggered": False,
                    "reason": "max_side_le_tile_threshold",
                },
            )
        return tile_heatmap_mosaic(
            adapter.heatmap_fn,
            image,
            attention_prompt,
            adapter.general_prompt,
            adapter.model,
            adapter.processor,
            tile_size=HIGH_RES_TILE_THRESHOLD,
        )
    raise ValueError(
        f"unsupported initial heatmap bootstrap: {args.initial_heatmap_bootstrap}"
    )


def compute_local_heatmap(
    *,
    crop: Image.Image,
    adapter: Any,
    attention_prompt: str,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any] | None]:
    """Compute one PHF local map, optionally preserving 1024-tile detail."""
    policy = str(args.local_heatmap_bootstrap)
    if policy == "full":
        return (
            adapter.heatmap_fn(
                crop,
                attention_prompt,
                adapter.general_prompt,
                adapter.model,
                adapter.processor,
            ),
            None,
        )
    if policy == "tile1024":
        if max(int(crop.width), int(crop.height)) <= HIGH_RES_TILE_THRESHOLD:
            return (
                adapter.heatmap_fn(
                    crop,
                    attention_prompt,
                    adapter.general_prompt,
                    adapter.model,
                    adapter.processor,
                ),
                {
                    "mode": "full_below_tile_threshold",
                    "tile_size": HIGH_RES_TILE_THRESHOLD,
                    "tile_count": 1,
                    "image_size": [int(crop.width), int(crop.height)],
                    "triggered": False,
                    "reason": "max_side_le_tile_threshold",
                },
            )
        return tile_heatmap_mosaic(
            adapter.heatmap_fn,
            crop,
            attention_prompt,
            adapter.general_prompt,
            adapter.model,
            adapter.processor,
            tile_size=HIGH_RES_TILE_THRESHOLD,
        )
    raise ValueError(f"unsupported local heatmap bootstrap: {policy}")


def run_sample(
    *,
    row: dict[str, Any],
    task: str,
    adapter: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    image = Image.open(image_path_for_row(task, row, args.image_root)).convert("RGB")
    question = str(row["question"])
    attention_prompt = adapter.attention_prompt(
        attention_question(row, str(args.attention_question_source))
    )
    answer_prompt = adapter.answer_prompt(question)

    with torch.no_grad():
        initial_heatmap, initial_heatmap_tile_info = compute_initial_heatmap(
            image=image,
            adapter=adapter,
            attention_prompt=attention_prompt,
            args=args,
        )
    initial_heatmap = sanitize_heatmap(initial_heatmap)
    initial_entropy = compute_normalized_heatmap_entropy(initial_heatmap)
    targets = build_stage_targets(
        image.size,
        min_crop_side=adapter.min_crop_side,
        initial_area_ratio=args.initial_area_ratio,
        area_ratio_decay=args.area_ratio_decay,
    )
    stop_threshold = getattr(args, "adaptive_stop_hotspot_ratio", ADAPTIVE_STOP_RATIO)
    local_heatmap_fn = (
        (lambda crop: compute_local_heatmap(
            crop=crop,
            adapter=adapter,
            attention_prompt=attention_prompt,
            args=args,
        ))
        if str(args.local_heatmap_bootstrap) == "tile1024"
        else None
    )
    stages = run_localization_chain(
        strategy=str(args.localization_strategy),
        image=image,
        adapter=adapter,
        attention_prompt=attention_prompt,
        initial_heatmap=initial_heatmap,
        targets=targets,
        topmass=float(args.topmass),
        adaptive_stop_hotspot_ratio=stop_threshold,
        local_heatmap_fn=local_heatmap_fn,
        stage_candidate_scorer=(
            partial(hotspot_component_square_bbox, component_score=args.component_score)
            if args.component_score != "default"
            else None
        ),
    )

    final_stage = stages[-1]
    final_bbox = bbox_to_int_tuple(final_stage["crop_bbox"])
    selected_ratio = component_short_ratio(final_stage)
    stop_reason = str(final_stage["stop_reason"])
    stop_info: dict[str, Any] = {
            "hotspot_ratio": stop_threshold,
            "ratio_side": "short",
            "mode": (
                "runtime_early_termination"
                if stop_threshold is not None
                else "sidecar_stop_disabled"
            ),
            "operator": ">=",
            "stop_reason": stop_reason,
            "final_bbox_semantics": "current_centered_square_crop",
            "selected_stage": final_stage.get("stage"),
            "selected_side": int(final_stage.get("side") or (final_bbox[2] - final_bbox[0])),
            "selected_target_label": final_stage.get("target_label"),
            "selected_component_ratio": float(selected_ratio),
            "selected_component_short_ratio": float(component_short_ratio(final_stage)),
            "stopped_by_rule": (
                stop_threshold is not None
                and stop_reason == "hotspot_fills_candidate"
            ),
            "executed_stage_count": len(stages),
            "stopped_before_local_heatmap": bool(
                final_stage.get("adaptive_stop_stopped_before_local_heatmap", False)
            ),
        }
    final_crop = image.crop(final_bbox)
    original_answer = adapter.original_answer_fn(
        image,
        answer_prompt,
        adapter.model,
        adapter.processor,
        int(args.max_new_tokens),
    )
    crop_answer = adapter.crop_answer_fn(
        image,
        final_crop,
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
            "bbox": list(final_bbox),
            "crop_method": str(args.method_name),
            "method_display_name": METHOD_DISPLAY_NAME,
            "method_chinese_name": METHOD_CN_NAME,
            "method_full_name": METHOD_FULL_NAME,
            "qpes_num_steps": len(stages),
            "qpes_localization_strategy": str(args.localization_strategy),
            "qpes_fusion_mode": (
                "sum_normalized_coverage_average"
                if str(args.localization_strategy) == "phf"
                else "none_last_local_only"
            ),
            "qpes_initial_heatmap_shape": [int(v) for v in initial_heatmap.shape],
            "qpes_bootstrap": str(args.initial_heatmap_bootstrap),
            "qpes_initial_heatmap_bootstrap": str(args.initial_heatmap_bootstrap),
            "qpes_local_heatmap_bootstrap": str(args.local_heatmap_bootstrap),
            "qpes_high_res_tile_threshold": (
                HIGH_RES_TILE_THRESHOLD
                if str(args.initial_heatmap_bootstrap) == "tile1024"
                else None
            ),
            "qpes_high_res_triggered": (
                bool(initial_heatmap_tile_info.get("triggered"))
                if isinstance(initial_heatmap_tile_info, dict)
                else False
            ),
            "qpes_initial_heatmap_tile_mosaic": jsonable(initial_heatmap_tile_info),
            "qpes_normalize_contributions": "sum",
            "qpes_selection_policy": "top1_hotspot_component_center",
            "qpes_topmass_fraction": float(args.topmass),
            "qpes_topmass_mask_covered_fractions": [
                float(stage.get("selection", {}).get("mask", {}).get("covered_fraction", 0.0))
                for stage in stages
            ],
            "qpes_attention_question_source": str(args.attention_question_source),
            "qpes_multistage_ratio_policy": args.stage_policy,
            "qpes_sensitivity_variant": getattr(args, "sensitivity_variant", None),
            "qpes_component_score": args.component_score,
            "qpes_multistage_ratios": [
                float(stage["target_ratio"])
                for stage in stages
                if stage["target_ratio"] is not None
            ],
            "qpes_target_labels": [stage["target_label"] for stage in stages],
            "qpes_target_sides": [int(stage["side"]) for stage in stages],
            "qpes_min_crop_side": int(adapter.min_crop_side),
            "qpes_heatmap_mode": adapter.heatmap_mode,
            "qpes_initial_heatmap_entropy": float(initial_entropy),
            "qpes_final_bbox": list(final_bbox),
            "qpes_final_bbox_semantics": "current_centered_square_crop",
            "qpes_stop_reason": stop_reason,
            "qpes_frozen_config_sha256": frozen_config_sha256(),
            "qpes_calibration_manifest_sha256": calibration_manifest_sha256(),
            "qpes_sensitivity_override": (
                {"topmass": float(args.topmass), "reason": str(args.sensitivity_reason)}
                if args.sensitivity_reason
                else None
            ),
            "qpes_final_input_policy": final_input_policy(
                adapter.name,
                adaptive_stop_enabled(args),
            ),
            "qpes_multistage_trace": [compact_stage(stage) for stage in stages],
        }
    )
    out_row["qpes_adaptive_stop"] = stop_info
    return out_row


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
    adapter: Any,
    last_sample_id: str | None = None,
    last_offset: int | None = None,
    resume_source: str | None = None,
) -> None:
    run_identity = manifest_identity(args, question_path, start_index, end_index)
    stop_threshold = getattr(args, "adaptive_stop_hotspot_ratio", None)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "execution_mode": "single",
        "status": status,
        "resume": bool(args.resume),
        "resume_source": resume_source,
        "run_identity": run_identity,
        "runtime_provenance": runtime_provenance(),
        "model": str(args.model),
        "task": str(args.task),
        "method": str(args.method_name),
        "method_display_name": METHOD_DISPLAY_NAME,
        "method_chinese_name": METHOD_CN_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "question_path": str(question_path),
        "range": [int(start_index), int(end_index)],
        "cap": int(args.cap) if args.cap is not None else None,
        "count": int(selected_count),
        "last_sample_id": last_sample_id,
        "last_offset": last_offset,
        "output": str(output_path),
        "completed_count": int(completed_count),
        "evaluation_report": str(path.parent / "evaluation_report.json"),
        "evaluation_csv": str(path.parent / "evaluation_report.csv"),
        "scorer_contract": str(path.parent / "scorer_contract.json"),
        "scorer_contract_version": SCORER_CONTRACT_VERSION,
        "gpu": args.gpu,
        "min_crop_side": int(adapter.min_crop_side),
        "stage_policy": args.stage_policy,
        "initial_area_ratio": args.initial_area_ratio,
        "ratio_decay": args.area_ratio_decay,
        "component_score": args.component_score,
        "normalize_contributions": "sum",
        "fusion": (
            "coverage_average"
            if str(args.localization_strategy) == "phf"
            else "none_last_local_only"
        ),
        "localization_strategy": str(args.localization_strategy),
        "bootstrap": str(args.initial_heatmap_bootstrap),
        "initial_heatmap_bootstrap": str(args.initial_heatmap_bootstrap),
        "high_res_tile_threshold": (
            HIGH_RES_TILE_THRESHOLD
            if str(args.initial_heatmap_bootstrap) == "tile1024"
            else None
        ),
        "selection_policy": "top1_hotspot_component_center",
        "topmass": float(args.topmass),
        "topmass_mask_policy": "threshold_ge_with_ties; actual covered fraction recorded per stage",
        "attention_question_source": str(args.attention_question_source),
        "frozen_config_sha256": frozen_config_sha256(),
        "calibration_manifest_sha256": calibration_manifest_sha256(),
        "sensitivity_override": (
            {"topmass": float(args.topmass), "reason": str(args.sensitivity_reason)}
            if args.sensitivity_reason
            else None
        ),
        "qwen_visual_tokens": (
            int(FROZEN_CONFIG["models"]["qwen2_5"]["visual_tokens"])
            if adapter.name == "qwen2_5"
            else None
        ),
        "qwen_max_pixels": (
            int(FROZEN_CONFIG["models"]["qwen2_5"]["max_pixels"])
            if adapter.name == "qwen2_5"
            else None
        ),
        "blip_bg_sub_alpha": float(args.blip_bg_sub_alpha),
        "heatmap_mode": adapter.heatmap_mode,
        "final_input_policy": final_input_policy(
            adapter.name,
            adaptive_stop_enabled(args),
        ),
        "adaptive_stop": {
            "hotspot_ratio": stop_threshold,
            "operator": ">=",
            "ratio_side": "short",
            "valid_component_required": True,
            "stop_reasons": ["hotspot_fills_candidate", "min_crop_side"],
            "final_bbox_semantics": "current_centered_square_crop",
            "execution": (
                "early_terminate_before_selected_stage_local_heatmap"
                if stop_threshold is not None
                else "stop_disabled_full_schedule"
            ),
        },
    }
    path.write_text(dumps_json(manifest), encoding="utf-8")


def run_single(args: argparse.Namespace) -> None:
    if args.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["PYTHONPATH"] = pythonpath_env()
    os.environ["QPES_BLIP_BG_SUB_ALPHA"] = str(args.blip_bg_sub_alpha)

    import torch

    question_path = question_path_for_task(str(args.task), args.question_path)
    all_rows = json.loads(question_path.read_text(encoding="utf-8"))
    cap_end = len(all_rows) if args.cap is None else min(int(args.cap), len(all_rows))
    end_index = cap_end if args.end_index is None else min(int(args.end_index), cap_end)
    selected_rows = all_rows[int(args.start_index) : int(end_index)]
    if not selected_rows:
        raise RuntimeError(f"no selected rows for task={args.task}")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_path = output_dir / f"{args.model}-{args.task}-{args.method_name}.json"
    manifest_path = output_dir / "run_manifest.json"
    expected_identity = manifest_identity(
        args,
        question_path,
        int(args.start_index),
        int(end_index),
    )
    resume_source = (
        validate_resume_manifest(manifest_path, expected_identity)
        if args.resume
        else None
    )
    ensure_output_contract(
        output_path,
        manifest_path,
        resume=bool(args.resume),
        allow_existing=bool(args.allow_existing_output),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_rows(output_path) if args.resume else {}
    outputs = [
        existing[str(row["id"])]
        for row in selected_rows
        if str(row.get("id")) in existing
    ]
    processed = {str(row.get("id")) for row in outputs}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = load_model_adapter(
        str(args.model),
        device,
        min_crop_side=args.min_crop_side,
        blip_bg_sub_alpha=float(args.blip_bg_sub_alpha),
    )
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
        adapter=adapter,
        resume_source=resume_source,
    )

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
            adapter=adapter,
            last_sample_id=sample_id,
            last_offset=int(offset),
            resume_source=resume_source,
        )
        should_log = (
            int(args.log_every) <= 1
            or len(outputs) == 1
            or len(outputs) == len(selected_rows)
            or len(outputs) % int(args.log_every) == 0
        )
        if should_log:
            print(
                f"[done] task={args.task} model={args.model} id={sample_id} "
                f"count={len(outputs)}/{len(selected_rows)}",
                flush=True,
            )

    ordered = {str(row["id"]): row for row in outputs}
    final_rows = [
        ordered[str(row["id"])]
        for row in selected_rows
        if str(row["id"]) in ordered
    ]
    if len(final_rows) != len(selected_rows):
        raise RuntimeError(
            f"output incomplete: got {len(final_rows)}, expected {len(selected_rows)}"
        )
    save_rows(output_path, final_rows)

    if args.evaluate:
        evaluate_output(output_dir, sys.executable)

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
        adapter=adapter,
        resume_source=resume_source,
    )


def main() -> None:
    args = apply_model_defaults(parse_args())
    if args.gpus:
        run_multigpu(args)
        return
    run_single(args)


if __name__ == "__main__":
    main()
