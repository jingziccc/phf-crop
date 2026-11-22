from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_repo(repo: Path | str = REPO_ROOT) -> Path:
    repo = Path(repo).resolve()
    os.chdir(repo)
    for path in (repo, repo / "code", repo / "code" / "runtime"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return repo


bootstrap_repo(REPO_ROOT)

from info import task_to_image_path, task_to_question_path  # noqa: E402
from top1_hotspot.adaptive_stop import component_short_ratio  # noqa: E402
from top1_hotspot.components import topmass_mask  # noqa: E402
from top1_hotspot.config import MODEL_DEFAULTS  # noqa: E402
from top1_hotspot.geometry import (  # noqa: E402
    bbox_to_int_tuple,
    build_stage_targets,
    image_bbox_to_heatmap_window,
    sanitize_heatmap,
)
from top1_hotspot.high_res import tile_heatmap_mosaic  # noqa: E402
from top1_hotspot.pipeline import (  # noqa: E402
    LOCALIZATION_STRATEGIES,
    run_localization_chain,
)


FINAL_INPUT_ORIGINAL_PLUS_CROP = "original_plus_final_crop"
FINAL_INPUT_BOXED_ORIGINAL = "boxed_original"
FINAL_INPUT_REDBOX_VITFLOOR_QUESTION = "boxed_original_plus_vitfloor_plus_question"

DEFAULT_CAMPAIGN_DIR = Path("outputs") / "final_7ds5000_hrbench_phf_crop"
CURRENT_SIDECAR_DIRS = (
    Path("outputs") / "llava_qwen_topmass_textvqa_vstar_hrbench_20260623",
    Path("outputs") / "blip_selected_param_vstar_hrbench_20260623",
)


@dataclass(frozen=True)
class VizConfig:
    repo: Path = REPO_ROOT
    model: str = "llava"
    task: str = "vstar"
    gpu: str = "4"
    sample_index: int = 5
    sample_id: str | None = None
    preview_start: int = 0
    preview_count: int = 8
    topmass: float | None = None
    vit_input_size: int | None = None
    blip_bg_sub_alpha: float | None = None
    qwen_bg_sub_alpha: float | None = 0.0
    attention_question_source: str = "short"
    initial_heatmap_bootstrap: str = "tile1024"
    adaptive_stop_hotspot_ratio: float | None = 1.0
    localization_strategy: str = "phf"
    max_new_tokens: int = 20
    answer_mode: str = "realtime"
    use_prediction_cache: bool = True
    final_input_mode: str = FINAL_INPUT_ORIGINAL_PLUS_CROP
    final_output_dir: Path | None = None
    save_dir: Path | None = None
    ppt_figsize: tuple[float, float] = (16, 9)
    save_dpi: int = 200

    def __post_init__(self) -> None:
        repo = Path(self.repo).resolve()
        object.__setattr__(self, "repo", repo)
        if self.final_output_dir is None:
            object.__setattr__(
                self,
                "final_output_dir",
                repo / DEFAULT_CAMPAIGN_DIR,
            )
        else:
            final_output_dir = Path(self.final_output_dir)
            if not final_output_dir.is_absolute():
                final_output_dir = repo / final_output_dir
            object.__setattr__(self, "final_output_dir", final_output_dir)
        if self.save_dir is None:
            object.__setattr__(self, "save_dir", repo / "outputs" / "visualization" / "multistage_trace")
        else:
            save_dir = Path(self.save_dir)
            if not save_dir.is_absolute():
                save_dir = repo / save_dir
            object.__setattr__(self, "save_dir", save_dir)


@dataclass
class VizContext:
    config: VizConfig
    run_index: dict[tuple[str, str], dict[str, Any]]
    available_tasks_by_model: dict[str, list[str]]
    run_info: dict[str, Any]
    rows: list[dict[str, Any]]
    question_path: Path
    predictions_by_id: dict[str, dict[str, Any]]
    prediction_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def prepare_environment(config: VizConfig) -> None:
    bootstrap_repo(config.repo)
    if config.gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTHONPATH", ".")
    os.environ["QPES_BLIP_BG_SUB_ALPHA"] = str(effective_blip_bg_sub_alpha(config))
    os.environ["QPES_QWEN_BG_SUB_ALPHA"] = str(effective_qwen_bg_sub_alpha(config))
    config.save_dir.mkdir(parents=True, exist_ok=True)


def model_defaults(model: str) -> dict[str, Any]:
    try:
        return dict(MODEL_DEFAULTS[str(model)])
    except KeyError as exc:
        raise ValueError(f"unsupported model for PHF-Crop visualization: {model!r}") from exc


def effective_topmass(config: VizConfig) -> float:
    if config.topmass is not None:
        return float(config.topmass)
    return float(model_defaults(config.model)["topmass"])


def effective_vit_input_size(config: VizConfig) -> int:
    if config.vit_input_size is not None:
        return int(config.vit_input_size)
    return int(model_defaults(config.model)["vit_input_size"])


def effective_blip_bg_sub_alpha(config: VizConfig) -> float:
    if config.blip_bg_sub_alpha is not None:
        return float(config.blip_bg_sub_alpha)
    return float(model_defaults(config.model).get("blip_bg_sub_alpha", 1.0))


def effective_qwen_bg_sub_alpha(config: VizConfig) -> float:
    return 0.0 if config.qwen_bg_sub_alpha is None else float(config.qwen_bg_sub_alpha)


def load_adapter_for_config(config: VizConfig):
    prepare_environment(config)
    import torch

    from top1_hotspot.adapters import load_model_adapter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = load_model_adapter(
        config.model,
        device,
        vit_input_size=effective_vit_input_size(config),
        blip_bg_sub_alpha=effective_blip_bg_sub_alpha(config),
    )
    return adapter, device


def normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", str(value)).strip()


def is_missing_answer(value: Any) -> bool:
    text = normalize_answer_text(value)
    return text == "" or text.lower() in {"not available", "n/a", "na", "none", "null"}


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_manifest_path(value: Any, *, repo: Path, manifest_path: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    for base in (repo, manifest_path.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return repo / path


def _candidate_manifest_paths(config: VizConfig) -> list[Path]:
    roots = [Path(config.final_output_dir)]
    roots.extend(config.repo / relative for relative in CURRENT_SIDECAR_DIRS)

    manifests: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*.manifest.json")) + sorted(root.rglob("run_manifest.json")):
            if "_shards" in manifest_path.parts:
                continue
            resolved = manifest_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            manifests.append(manifest_path)
    return manifests


def _default_method_name(model: str) -> str:
    return str(model_defaults(model).get("method_name", ""))


def _float_matches(value: Any, expected: float | None, tol: float = 1e-9) -> bool:
    if value is None or expected is None:
        return False
    try:
        return abs(float(value) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def _manifest_score(manifest: dict[str, Any]) -> float:
    model = str(manifest.get("model") or "")
    method = str(manifest.get("method") or "")
    score = 0.0
    if str(manifest.get("status") or "").lower() == "complete":
        score += 30.0
    if model in MODEL_DEFAULTS and method and method == _default_method_name(model):
        score += 80.0

    defaults = model_defaults(model) if model in MODEL_DEFAULTS else {}
    if _float_matches(manifest.get("topmass"), defaults.get("topmass")):
        score += 60.0
    if _float_matches(manifest.get("vit_input_size"), defaults.get("vit_input_size")):
        score += 20.0
    if model == "blip" and _float_matches(
        manifest.get("blip_bg_sub_alpha"),
        defaults.get("blip_bg_sub_alpha", 1.0),
    ):
        score += 60.0
    if "halveto_vit" in method or "halve" in str(manifest.get("stage_policy") or ""):
        score += 20.0
    return score


def _manifest_to_run_info(config: VizConfig, manifest_path: Path) -> dict[str, Any] | None:
    manifest = read_json(manifest_path)
    model = str(manifest.get("model") or "").strip()
    task = str(manifest.get("task") or "").strip()
    output = manifest.get("output")
    if not model or not task or not output:
        return None
    output_path = _resolve_manifest_path(output, repo=config.repo, manifest_path=manifest_path)
    question_value = manifest.get("question_path") or task_to_question_path.get(task)
    question_path = _resolve_manifest_path(question_value, repo=config.repo, manifest_path=manifest_path)
    return {
        "model": model,
        "task": task,
        "method": manifest.get("method"),
        "count": manifest.get("count") or manifest.get("completed_count"),
        "status": manifest.get("status"),
        "manifest_path": manifest_path,
        "prediction_path": output_path,
        "question_path": question_path,
        "score": _manifest_score(manifest),
        "topmass": manifest.get("topmass"),
        "vit_input_size": manifest.get("vit_input_size"),
        "blip_bg_sub_alpha": manifest.get("blip_bg_sub_alpha"),
    }


def discover_run_index(config: VizConfig) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in _candidate_manifest_paths(config):
        info = _manifest_to_run_info(config, manifest_path)
        if info is None:
            continue
        key = (info["model"], info["task"])
        previous = index.get(key)
        if previous is None or float(info["score"]) > float(previous.get("score", -1.0)):
            index[key] = info
    for info in index.values():
        info.pop("score", None)
    return index


def available_tasks_by_model(run_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for model, task in sorted(run_index):
        available.setdefault(model, []).append(task)
    return available


def run_info_for(
    run_index: dict[tuple[str, str], dict[str, Any]],
    available: dict[str, list[str]],
    model: str,
    task: str,
) -> dict[str, Any]:
    info = run_index.get((model, task))
    if info is not None:
        return info
    model_tasks = available.get(model, [])
    if model_tasks:
        raise ValueError(f"No prediction manifest for model={model!r}, task={task!r}. Available tasks: {model_tasks}")
    raise ValueError(f"No prediction manifest for model={model!r}. Available models: {sorted(available)}")


def question_path_for(config: VizConfig, run_index: dict[tuple[str, str], dict[str, Any]]) -> Path:
    info = run_index.get((config.model, config.task), {})
    path = info.get("question_path")
    if path and Path(path).exists():
        return Path(path)
    if config.task in task_to_question_path:
        return Path(task_to_question_path[config.task])
    fallback = config.repo / "data" / config.task / "data.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No question path found for task={config.task!r}")


def image_root_for(config: VizConfig, question_path: Path | None = None) -> Path:
    if config.task in task_to_image_path:
        return Path(task_to_image_path[config.task])
    if question_path is not None:
        candidate = Path(question_path).parent / "images"
        if candidate.exists():
            return candidate
    return config.repo / "data" / config.task / "images"


def image_path_for_row(config: VizConfig, row: dict[str, Any], question_path: Path | None = None) -> Path:
    image_root = image_root_for(config, question_path=question_path)
    if "image_path" in row and row["image_path"] is not None:
        image_path = Path(str(row["image_path"]))
        return image_path if image_path.is_absolute() else image_root / image_path
    if "image_id" in row and row["image_id"] is not None:
        return image_root / f"{row['image_id']}.jpg"
    raise KeyError("row must contain image_path or image_id")


def original_image_path(context: VizContext, row: dict[str, Any]) -> Path:
    return image_path_for_row(context.config, row, question_path=context.question_path).resolve()


def original_image_markdown(context: VizContext, row: dict[str, Any], label: str = "Open original image") -> str:
    source_path = image_path_for_row(context.config, row, question_path=context.question_path)
    source_path = source_path if source_path.is_absolute() else context.config.repo / source_path
    resolved_path = source_path.resolve()
    status = "" if resolved_path.exists() else " missing"
    notebook_dir = context.config.repo / "notebooks"
    try:
        link_target = os.path.relpath(source_path, notebook_dir)
    except ValueError:
        link_target = str(source_path)
    link_target = Path(link_target).as_posix()
    return f"[{label}]({link_target}) `{source_path}`{status}"


def display_original_image_link(context: VizContext, row: dict[str, Any], label: str = "Open original image") -> str:
    markdown = original_image_markdown(context, row, label=label)
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
    except Exception:
        print(markdown)
        return markdown

    if get_ipython() is None:
        print(markdown)
    else:
        display(Markdown(markdown))
    return markdown


def attention_question(row: dict[str, Any], source: str) -> str:
    if source == "short":
        return str(row.get("short_question") or row["question"])
    if source == "full":
        return str(row["question"])
    raise ValueError(f"unsupported attention question source: {source}")


def mcq_choices_from_question(question: Any) -> dict[str, str]:
    choices: dict[str, str] = {}
    for line in str(question).splitlines():
        match = re.match(r"^\s*(?:\(([A-D])\)|([A-D])[\.)])\s+(.+?)\s*$", line)
        if not match:
            continue
        key = (match.group(1) or match.group(2)).upper()
        value = match.group(3).strip()
        if value and not value.lower().startswith("answer with"):
            choices[key] = value
    return choices


def mcq_choices_for_row(row: dict[str, Any]) -> dict[str, str]:
    choices = row.get("mcq_choices")
    if isinstance(choices, dict):
        return {str(key).upper(): str(value) for key, value in choices.items()}
    return mcq_choices_from_question(row.get("question", ""))


def format_label_value(value: Any, row: dict[str, Any] | None = None, max_unique: int = 6) -> str:
    row = row or {}
    choices = mcq_choices_for_row(row)
    if isinstance(value, (list, tuple)):
        texts = [normalize_answer_text(item) for item in value]
        texts = [text for text in texts if text]
        if not texts:
            return "not available"
        counts = Counter(texts)
        pieces = []
        for text, count in counts.most_common(max_unique):
            suffix = f" x{count}" if count > 1 else ""
            pieces.append(f"{text}{suffix}")
        if len(counts) > max_unique:
            pieces.append("...")
        return ", ".join(pieces)

    text = normalize_answer_text(value)
    if not text:
        return "not available"
    key = text.upper()
    if key in choices:
        return f"{key} ({choices[key]})"
    return text


def sample_answer_text(row: dict[str, Any]) -> str:
    for key in ("labels", "label", "answer", "gt_answer", "ground_truth", "target"):
        if key in row and not is_missing_answer(row.get(key)):
            return format_label_value(row.get(key), row=row)
    return "not available"


def _answer_label_value(row: dict[str, Any]) -> Any:
    for key in ("labels", "label", "answer", "gt_answer", "ground_truth", "target"):
        if key in row and not is_missing_answer(row.get(key)):
            return row.get(key)
    return None


def _labels_as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [normalize_answer_text(item) for item in value if normalize_answer_text(item)]
    text = normalize_answer_text(value)
    return [text] if text else []


def _mcq_label(row: dict[str, Any]) -> str:
    text = normalize_answer_text(_answer_label_value(row)).upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return text[:1] if text[:1] in "ABCD" else text


def _mcq_candidates(row: dict[str, Any]) -> list[str]:
    question = str(row.get("question", ""))
    candidates = [line.strip() for line in question.splitlines()[1:-1] if line.strip()]
    if candidates:
        return candidates
    choices = mcq_choices_for_row(row)
    return [f"({key}) {choices[key]}" for key in sorted(choices)]


def single_answer_score(task: str, row: dict[str, Any], answer: Any) -> float | None:
    if is_missing_answer(answer):
        return None
    labels = _answer_label_value(row)
    if labels is None:
        return None

    answer_text = normalize_answer_text(answer)
    if task in {"vstar", "hrbench_4k", "hrbench_8k"}:
        from get_score import get_index, normalize_mcq_letter

        candidates = _mcq_candidates(row)
        pred_letter = normalize_mcq_letter(answer_text)
        if pred_letter not in "ABCD" and candidates:
            pred_letter = "ABCD"[get_index(answer_text, candidates)]
        return 100.0 if pred_letter == _mcq_label(row) else 0.0

    if task == "pope":
        from get_score import match_mcq

        label = normalize_answer_text(labels).lower()
        pred = match_mcq(answer_text.lower(), ["yes", "no"])
        return 100.0 if pred == label else 0.0

    label_list = _labels_as_list(labels)
    if not label_list:
        return None

    if task in {"textvqa", "aokvqa", "vqav2"}:
        from get_score import get_acc

        return float(get_acc(answer_text, label_list))
    if task == "gqa":
        from get_score import get_acc_gqa

        return float(get_acc_gqa(answer_text, label_list))
    if task == "docvqa":
        return 100.0 if any(answer_text in label or label in answer_text for label in label_list) else 0.0
    return None


def format_sample_score(score: float | int | None) -> str:
    if score is None:
        return "not available"
    score = float(score)
    if abs(score - round(score)) < 1e-9:
        return f"{int(round(score))}/100"
    return f"{score:.2f}/100"


def load_prediction_lookup(config: VizConfig, run_index: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Path | None]:
    info = run_index.get((config.model, config.task))
    path = info.get("prediction_path") if info else None
    if path is None or not Path(path).exists():
        return {}, path
    pred_rows = read_json(path)
    lookup: dict[str, dict[str, Any]] = {}
    for item in pred_rows:
        if "id" in item:
            lookup[str(item["id"])] = item
    return lookup, Path(path)


def display_answer(value: Any, limit: int = 90) -> str:
    text = normalize_answer_text(value)
    if not text:
        return "not available"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_dataset(config: VizConfig, run_index: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], Path]:
    qpath = question_path_for(config, run_index)
    return read_json(qpath), qpath


def choose_sample(rows: list[dict[str, Any]], sample_id: str | None = None, sample_index: int = 0) -> dict[str, Any]:
    if sample_id is not None and str(sample_id).strip() != "":
        wanted = str(sample_id)
        for row in rows:
            if str(row.get("id")) == wanted:
                return row
        raise KeyError(f"missing sample id: {wanted}")
    return rows[int(sample_index)]


def prediction_for_row(context: VizContext, row: dict[str, Any]) -> dict[str, Any]:
    return context.predictions_by_id.get(str(row.get("id")), {})


def preview_samples(context: VizContext, start: int | None = None, count: int | None = None) -> list[dict[str, Any]]:
    start = context.config.preview_start if start is None else int(start)
    count = context.config.preview_count if count is None else int(count)
    preview = []
    for index, row in enumerate(context.rows[start : start + count], start=start):
        prediction = prediction_for_row(context, row)
        image_path = original_image_path(context, row)
        preview.append(
            {
                "index": index,
                "id": row.get("id"),
                "answer": sample_answer_text({**prediction, **row}),
                "has_prediction": bool(prediction),
                "original_answer": display_answer(prediction.get("original_answer"), 60),
                "crop_answer": display_answer(prediction.get("crop_answer"), 60),
                "image_exists": image_path.exists(),
                "image_path": str(image_path),
                "question": str(row.get("question", "")).splitlines()[0][:120],
            }
        )
    return preview


def build_context(config: VizConfig) -> VizContext:
    prepare_environment(config)
    run_index = discover_run_index(config)
    available = available_tasks_by_model(run_index)
    try:
        run_info = run_info_for(run_index, available, config.model, config.task)
    except ValueError:
        run_info = {
            "model": config.model,
            "task": config.task,
            "method": None,
            "count": None,
            "status": "dataset_only",
            "manifest_path": None,
            "prediction_path": None,
            "question_path": None,
        }
    rows, question_path = load_dataset(config, run_index)
    if config.use_prediction_cache:
        predictions_by_id, prediction_path = load_prediction_lookup(config, run_index)
    else:
        predictions_by_id, prediction_path = {}, None
    return VizContext(
        config=config,
        run_index=run_index,
        available_tasks_by_model=available,
        run_info=run_info,
        rows=rows,
        question_path=question_path,
        predictions_by_id=predictions_by_id,
        prediction_path=prediction_path,
    )


def answer_source(value: Any, prediction: dict[str, Any], key: str) -> str:
    if not is_missing_answer(value) and key in prediction:
        return "prediction"
    if not is_missing_answer(value):
        return "dataset"
    return "missing"


def maybe_generate_answers(
    *,
    adapter: Any,
    config: VizConfig,
    image: Image.Image,
    final_crop: Image.Image,
    final_input_image: Image.Image | None = None,
    final_input_images: list[tuple[str, Image.Image]] | None = None,
    question: str,
    prediction: dict[str, Any],
) -> tuple[str, str, str, str]:
    answer_prompt = adapter.answer_prompt(question)
    if config.answer_mode == "prediction_then_realtime":
        original_answer = prediction.get("original_answer")
        crop_answer = prediction.get("crop_answer")
        original_source = answer_source(original_answer, prediction, "original_answer")
        crop_source = answer_source(crop_answer, prediction, "crop_answer")
        if config.final_input_mode != FINAL_INPUT_ORIGINAL_PLUS_CROP:
            crop_answer = None
            crop_source = "missing"
    elif config.answer_mode == "realtime":
        original_answer = None
        crop_answer = None
        original_source = "missing"
        crop_source = "missing"
    else:
        raise ValueError(f"unsupported answer_mode: {config.answer_mode!r}")

    if config.answer_mode == "realtime" or is_missing_answer(original_answer):
        print("generating original_answer in real time...")
        original_answer = adapter.original_answer_fn(
            image,
            answer_prompt,
            adapter.model,
            adapter.processor,
            int(config.max_new_tokens),
        )
        original_source = "realtime"

    if config.answer_mode == "realtime" or is_missing_answer(crop_answer):
        if config.final_input_mode == FINAL_INPUT_BOXED_ORIGINAL:
            if final_input_image is None:
                raise ValueError("final_input_image is required when final_input_mode='boxed_original'")
            print("generating crop_answer from boxed original image in real time...")
            crop_answer = adapter.original_answer_fn(
                final_input_image,
                answer_prompt,
                adapter.model,
                adapter.processor,
                int(config.max_new_tokens),
            )
            crop_source = "realtime_boxed_original"
        elif config.final_input_mode == FINAL_INPUT_REDBOX_VITFLOOR_QUESTION:
            if not final_input_images:
                raise ValueError(
                    "final_input_images is required when "
                    f"final_input_mode={FINAL_INPUT_REDBOX_VITFLOOR_QUESTION!r}"
                )
            print("generating crop_answer from red-box original + ViT floor + original question in real time...")
            crop_answer = answer_from_image_sequence(
                adapter=adapter,
                images=[item[1] for item in final_input_images],
                prompt=answer_prompt,
                max_new_tokens=int(config.max_new_tokens),
            )
            crop_source = "realtime_redbox_vitfloor_question"
        elif config.final_input_mode == FINAL_INPUT_ORIGINAL_PLUS_CROP:
            print("generating crop_answer in real time...")
            crop_answer = adapter.crop_answer_fn(
                image,
                final_crop,
                answer_prompt,
                adapter.model,
                adapter.processor,
                int(config.max_new_tokens),
            )
            crop_source = "realtime"
        else:
            raise ValueError(f"unsupported final_input_mode: {config.final_input_mode!r}")

    return str(original_answer), str(crop_answer), original_source, crop_source


def answer_from_image_sequence(
    *,
    adapter: Any,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> str:
    if adapter.name == "llava":
        return llava_answer_from_images(
            images=images,
            prompt=prompt,
            model=adapter.model,
            processor=adapter.processor,
            max_new_tokens=max_new_tokens,
        )
    if adapter.name == "qwen2_5":
        return qwen_answer_from_images(
            images=images,
            prompt=prompt,
            model=adapter.model,
            processor=adapter.processor,
            max_new_tokens=max_new_tokens,
        )
    if adapter.name == "blip":
        context_image = compose_image_strip(images, target_height=adapter.vit_input_size, label_height=0)
        return adapter.original_answer_fn(
            context_image,
            prompt,
            adapter.model,
            adapter.processor,
            max_new_tokens,
        )
    raise ValueError(f"unsupported adapter for image sequence: {adapter.name!r}")


def llava_answer_from_images(
    *,
    images: list[Image.Image],
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    from top1_hotspot.adapters import _decode_llava

    image_tokens = "<image>" * len(images)
    if "<image>\n" in prompt:
        multi_prompt = prompt.replace("<image>\n", f"{image_tokens}\n", 1)
    else:
        multi_prompt = f"{image_tokens}\n{prompt}"
    inputs = processor(text=multi_prompt, images=images, return_tensors="pt", padding=True).to(
        model.device,
        torch.bfloat16,
    )
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = _decode_llava(processor, generate_ids)
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def qwen_answer_from_images(
    *,
    images: list[Image.Image],
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> str:
    import torch

    from utils import encode_base64, prepare_qwen2_5_input

    content = [
        {"type": "image", "image": f"data:image;base64,{encode_base64(input_image)}"}
        for input_image in images
    ]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generate_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def final_top1_cluster_bbox(stages: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
    if not stages:
        return None
    selected = stages[-1].get("selection", {}).get("selected_component")
    if not selected or "image_bbox" not in selected:
        return None
    return bbox_to_int_tuple(selected["image_bbox"])


def compute_visual_initial_heatmap(
    *,
    image: Image.Image,
    adapter: Any,
    attention_prompt: str,
    config: VizConfig,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    import torch

    mode = str(config.initial_heatmap_bootstrap)
    with torch.no_grad():
        if mode == "full":
            heatmap = adapter.heatmap_fn(
                image,
                attention_prompt,
                adapter.general_prompt,
                adapter.model,
                adapter.processor,
            )
            return sanitize_heatmap(heatmap), None
        if mode == "tile1024":
            return tile_heatmap_mosaic(
                adapter.heatmap_fn,
                image,
                attention_prompt,
                adapter.general_prompt,
                adapter.model,
                adapter.processor,
                tile_size=1024,
            )
    raise ValueError(f"unsupported initial_heatmap_bootstrap: {mode!r}")


def draw_original_with_box(
    image: Image.Image,
    bbox: Any,
    *,
    color: str = "red",
    width: int | None = None,
) -> Image.Image:
    boxed = image.copy()
    if width is None:
        width = max(4, int(round(min(image.size) / 220.0)))
    draw = ImageDraw.Draw(boxed)
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    for offset in range(int(width)):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
    return boxed


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    height = max(1, int(height))
    scale = height / max(1, image.size[1])
    width = max(1, int(round(image.size[0] * scale)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling)


def compose_image_strip(
    images: list[Image.Image],
    *,
    labels: list[str] | None = None,
    target_height: int = 720,
    label_height: int = 42,
    gap: int = 12,
) -> Image.Image:
    panels = [_resize_to_height(input_image.convert("RGB"), target_height) for input_image in images]
    total_width = sum(panel.size[0] for panel in panels) + max(0, len(panels) - 1) * gap
    total_height = target_height + max(0, int(label_height))
    strip = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(strip)
    x = 0
    labels = labels or []
    for idx, panel in enumerate(panels):
        strip.paste(panel, (x, label_height))
        draw.rectangle((x, label_height, x + panel.size[0] - 1, total_height - 1), outline="#444444", width=2)
        if label_height > 0 and idx < len(labels):
            draw.text((x + 8, 8), labels[idx], fill="#111111")
        x += panel.size[0] + gap
    return strip


def run_visual_trace(
    row: dict[str, Any],
    *,
    context: VizContext,
    adapter: Any,
    generate_answers: bool = True,
) -> dict[str, Any]:
    config = context.config
    image_path = original_image_path(context, row)
    image = Image.open(image_path).convert("RGB")
    question = str(row["question"])
    prediction = prediction_for_row(context, row)
    truth_row = {**prediction, **row}
    att_q = attention_question(truth_row, config.attention_question_source)
    attention_prompt = adapter.attention_prompt(att_q)

    initial_heatmap, initial_heatmap_tile_info = compute_visual_initial_heatmap(
        image=image,
        adapter=adapter,
        attention_prompt=attention_prompt,
        config=config,
    )
    initial_heatmap = sanitize_heatmap(initial_heatmap)
    targets = build_stage_targets(image.size, min_crop_side=adapter.min_crop_side)
    if config.localization_strategy not in LOCALIZATION_STRATEGIES:
        raise ValueError(
            f"unsupported localization strategy: {config.localization_strategy!r}; "
            f"expected one of {LOCALIZATION_STRATEGIES}"
        )
    runtime_stop_ratio = (
        float(config.adaptive_stop_hotspot_ratio)
        if config.adaptive_stop_hotspot_ratio is not None
        else float("inf")
    )
    stages = run_localization_chain(
        strategy=config.localization_strategy,
        image=image,
        adapter=adapter,
        attention_prompt=attention_prompt,
        initial_heatmap=initial_heatmap,
        targets=targets,
        topmass=effective_topmass(config),
        adaptive_stop_hotspot_ratio=runtime_stop_ratio,
    )
    final_stage = stages[-1]
    adaptive_stop_info: dict[str, Any] | None = None
    if config.adaptive_stop_hotspot_ratio is not None:
        adaptive_stop_info = {
            "hotspot_ratio": float(config.adaptive_stop_hotspot_ratio),
            "ratio_side": "short",
            "mode": "runtime_early_termination",
            "stop_reason": final_stage.get("stop_reason"),
            "selected_stage": final_stage.get("stage"),
            "selected_side": int(final_stage.get("side") or 0),
            "selected_target_label": final_stage.get("target_label"),
            "selected_component_ratio": float(component_short_ratio(final_stage)),
            "selected_component_short_ratio": float(component_short_ratio(final_stage)),
            "stopped_by_rule": final_stage.get("stop_reason") == "hotspot_fills_candidate",
            "stopped_before_local_heatmap": bool(
                final_stage.get("adaptive_stop_stopped_before_local_heatmap", False)
            ),
        }
    final_bbox = bbox_to_int_tuple(final_stage["crop_bbox"])
    final_crop = image.crop(final_bbox)
    selected_final_component = final_stage.get("selection", {}).get("selected_component")
    final_cluster_bbox = (
        bbox_to_int_tuple(selected_final_component["image_bbox"])
        if isinstance(selected_final_component, dict)
        and selected_final_component.get("image_bbox") is not None
        else final_top1_cluster_bbox(stages)
    )
    final_input_images: list[tuple[str, Image.Image]] = [
        ("original", image),
        ("vitfloor_crop", final_crop),
    ]
    final_input_image = final_crop
    final_input_label = "original_plus_final_crop"
    if config.final_input_mode == FINAL_INPUT_BOXED_ORIGINAL:
        if final_cluster_bbox is None:
            raise ValueError("final top1 cluster bbox is missing")
        final_input_image = draw_original_with_box(image, final_cluster_bbox, color="red")
        final_input_label = "boxed_original_top1_cluster"
        final_input_images = [("redbox_original", final_input_image)]
    elif config.final_input_mode == FINAL_INPUT_REDBOX_VITFLOOR_QUESTION:
        if final_cluster_bbox is None:
            raise ValueError("final top1 cluster bbox is missing")
        boxed_original = draw_original_with_box(image, final_cluster_bbox, color="red")
        final_input_images = [
            ("redbox_original", boxed_original),
            ("vitfloor_crop", final_crop),
        ]
        final_input_image = compose_image_strip(
            [item[1] for item in final_input_images],
            labels=["red-box original", "ViT floor crop"],
            target_height=720,
        )
        final_input_label = "redbox_original_plus_vitfloor_plus_question"
    elif config.final_input_mode != FINAL_INPUT_ORIGINAL_PLUS_CROP:
        raise ValueError(f"unsupported final_input_mode: {config.final_input_mode!r}")
    if generate_answers:
        original_answer, crop_answer, original_source, crop_source = maybe_generate_answers(
            adapter=adapter,
            config=config,
            image=image,
            final_crop=final_crop,
            final_input_image=final_input_image,
            final_input_images=final_input_images,
            question=question,
            prediction=prediction,
        )
    else:
        original_answer = prediction.get("original_answer")
        crop_answer = prediction.get("crop_answer")
        original_source = answer_source(original_answer, prediction, "original_answer")
        crop_source = answer_source(crop_answer, prediction, "crop_answer")
        if config.final_input_mode != FINAL_INPUT_ORIGINAL_PLUS_CROP:
            crop_answer = None
            crop_source = "missing"
    score_row = {**prediction, **row}
    original_score = single_answer_score(config.task, score_row, original_answer)
    crop_score = single_answer_score(config.task, score_row, crop_answer)
    return {
        "row": row,
        "prediction": prediction,
        "image": image,
        "question": question,
        "attention_question": att_q,
        "answer_text": sample_answer_text(truth_row),
        "original_answer": original_answer,
        "crop_answer": crop_answer,
        "original_answer_source": original_source,
        "crop_answer_source": crop_source,
        "original_score": original_score,
        "crop_score": crop_score,
        "question_path": str(context.question_path),
        "prediction_path": str(context.prediction_path) if context.prediction_path else None,
        "image_path": str(image_path),
        "final_input_mode": config.final_input_mode,
        "final_input_label": final_input_label,
        "final_input_image": final_input_image,
        "final_input_images": final_input_images,
        "final_input_prompt": adapter.answer_prompt(question),
        "final_top1_cluster_bbox": final_cluster_bbox,
        "final_bbox": final_bbox,
        "final_stage": final_stage,
        "adaptive_stop": adaptive_stop_info,
        "initial_heatmap": initial_heatmap,
        "initial_heatmap_bootstrap": config.initial_heatmap_bootstrap,
        "localization_strategy": config.localization_strategy,
        "initial_heatmap_tile_info": initial_heatmap_tile_info,
        "targets": targets,
        "stages": stages,
    }


def heatmap_for_display(heatmap: Any) -> np.ndarray:
    array = sanitize_heatmap(heatmap).astype(np.float32)
    lo = float(array.min())
    hi = float(array.max())
    if hi > lo:
        array = (array - lo) / (hi - lo)
    return array


def show_heatmap_overlay(ax: Any, image: Image.Image, heatmap: Any, alpha: float = 0.45) -> None:
    array = heatmap_for_display(heatmap)
    ax.imshow(image, extent=(0, image.size[0], image.size[1], 0))
    ax.imshow(
        array,
        cmap="inferno",
        alpha=alpha,
        extent=(0, image.size[0], image.size[1], 0),
        interpolation="bicubic",
    )
    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)
    ax.axis("off")


def draw_rect(ax: Any, bbox: Any, color: str, label: str | None = None, linewidth: float = 2.0, linestyle: str = "-") -> None:
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    rect = patches.Rectangle(
        (x1, y1),
        max(1, x2 - x1),
        max(1, y2 - y1),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(rect)


def draw_heatmap_rect(
    ax: Any,
    bbox: Any,
    color: str,
    label: str | None = None,
    linewidth: float = 2.0,
    linestyle: str = "-",
) -> None:
    draw_rect(ax, bbox, color, label=label, linewidth=linewidth, linestyle=linestyle)


def plot_tile1024_initial_heatmap(
    trace: dict[str, Any],
    *,
    context: VizContext,
    save: bool = False,
    figsize: tuple[float, float] | None = None,
):
    tile_info = trace.get("initial_heatmap_tile_info")
    if not tile_info:
        raise ValueError(
            "trace has no tile heatmap metadata; set "
            "VizConfig(initial_heatmap_bootstrap='tile1024')"
        )

    image = trace["image"]
    heatmap = trace["initial_heatmap"]
    row = trace["row"]
    sample_id = str(row.get("id", "unknown"))
    stages = trace.get("stages") or []
    n_stages = max(1, len(stages))
    n_cols = max(5, n_stages)
    if figsize is None:
        figsize = (max(20.0, 3.2 * n_cols), 12.8)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=[1.35, 1.75, 1.2],
        left=0.04,
        right=0.985,
        bottom=0.055,
        top=0.875,
        hspace=0.28,
        wspace=0.10,
    )
    title = wrap_text(question_prompt_text(row), width=124, limit=430)
    fig.suptitle(
        f"{context.config.model} / {context.config.task} / sample {sample_id} / {tile_info['mode']}\n{title}",
        fontsize=10.8,
        fontweight="semibold",
        y=0.975,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image, extent=(0, image.size[0], image.size[1], 0))
    for tile in tile_info.get("tile_boxes", []):
        draw_rect(ax, tile["bbox"], "#00d1ff", linewidth=1.25)
    ax.set_title("Original image tile grid", fontsize=9.5)
    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(heatmap_for_display(heatmap), cmap="inferno", interpolation="nearest", extent=heatmap_extent(heatmap))
    heatmap_h, heatmap_w = sanitize_heatmap(heatmap).shape
    grid_cols = int(tile_info["grid_cols"])
    grid_rows = int(tile_info["grid_rows"])
    for col in range(1, grid_cols):
        x = heatmap_w * col / grid_cols
        ax.axvline(x, color="#00d1ff", linewidth=1.0, alpha=0.75)
    for row_idx in range(1, grid_rows):
        y = heatmap_h * row_idx / grid_rows
        ax.axhline(y, color="#00d1ff", linewidth=1.0, alpha=0.75)
    if stages:
        stage0_window = image_bbox_to_heatmap_window(
            stages[0]["crop_bbox"],
            image_size=image.size,
            heatmap_shape=sanitize_heatmap(heatmap).shape,
        )
        draw_heatmap_rect(ax, stage0_window, "#00d1ff", linewidth=2.0)
    ax.set_title(f"Stitched tile heatmap ({heatmap_w}x{heatmap_h})", fontsize=9.5)
    ax.set_xlim(0, heatmap_w)
    ax.set_ylim(heatmap_h, 0)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2:4])
    ax.axis("off")
    if stages:
        final_bbox_for_panel = bbox_to_int_tuple(trace.get("final_bbox") or stages[-1]["crop_bbox"])
        final_crop_for_panel = image.crop(final_bbox_for_panel)
        ax.imshow(
            final_crop_for_panel,
            extent=(0, final_crop_for_panel.size[0], final_crop_for_panel.size[1], 0),
        )
        ax.set_xlim(0, final_crop_for_panel.size[0])
        ax.set_ylim(final_crop_for_panel.size[1], 0)
        ax.set_title(
            f"Final crop image\n{final_crop_for_panel.size[0]}x{final_crop_for_panel.size[1]}",
            fontsize=9.5,
        )
    else:
        ax.text(0.5, 0.5, "No final crop", ha="center", va="center", fontsize=10)

    ax = fig.add_subplot(gs[0, 4:])
    ax.axis("off")
    tile_sizes = sorted({tuple(tile["tile_size"]) for tile in tile_info.get("tile_boxes", [])})
    stop_info = trace.get("adaptive_stop")
    if isinstance(stop_info, dict):
        stop_text = (
            f"S{stop_info['selected_stage']} {stop_info['selected_target_label']} "
            f"short-ratio={stop_info['selected_component_short_ratio']:.3g} "
            f"threshold={stop_info['hotspot_ratio']:.3g} "
            f"stopped={stop_info['stopped_by_rule']}"
        )
    else:
        stop_text = "disabled; final ViT-floor stage"
    summary = [
        f"Ground truth: {display_answer(trace.get('answer_text'), 120)}",
        f"Original answer: {display_answer(trace.get('original_answer'), 120)}",
        f"Original score: {format_sample_score(trace.get('original_score'))}",
        f"Crop answer: {display_answer(trace.get('crop_answer'), 120)}",
        f"Crop score: {format_sample_score(trace.get('crop_score'))}",
        "",
        f"Bootstrap: {tile_info['mode']}",
        f"Normalization: {tile_info['normalization']}",
        f"Image size: {tuple(tile_info['image_size'])}",
        f"Tile threshold: {tile_info['tile_size']}",
        f"Grid: {tile_info['grid_cols']} x {tile_info['grid_rows']}",
        f"Tile count: {tile_info['tile_count']}",
        f"Step size: {tuple(tile_info['step_size'])}",
        f"Tile sizes: {tile_sizes[:6]}{' ...' if len(tile_sizes) > 6 else ''}",
        f"Mosaic heatmap shape: {tuple(tile_info['mosaic_heatmap_shape'])}",
        f"Initial entropy: {trace.get('stages', [{}])[0].get('pre_entropy', 'not available') if stages else 'not available'}",
        f"Adaptive stop: {stop_text}",
    ]
    summary.extend(
        [
            f"PHF stages: {len(stages)}",
            "Overview row: original image with all stage boxes and final crop.",
            "Stage strip: crop image with crop-local heatmap overlay.",
        ]
    )
    ax.text(0.0, 1.0, "\n".join(summary), va="top", ha="left", fontsize=9.5, linespacing=1.34)

    ax_overview = fig.add_subplot(gs[1, :])
    ax_overview.imshow(image, extent=(0, image.size[0], image.size[1], 0))
    stage_colors = [
        "#00d1ff",
        "#ffb000",
        "#7bd88f",
        "#b967ff",
        "#ff6b6b",
        "#00a8a8",
        "#f28e2b",
        "#59a14f",
    ]
    for index, stage in enumerate(stages):
        color = stage_colors[index % len(stage_colors)]
        crop_bbox = bbox_to_int_tuple(stage["crop_bbox"])
        is_stop_stage = (
            isinstance(stop_info, dict)
            and int(stop_info.get("selected_stage") or -1) == int(stage["stage"])
        )
        is_final = crop_bbox == bbox_to_int_tuple(trace.get("final_bbox") or stages[-1]["crop_bbox"])
        draw_rect(
            ax_overview,
            crop_bbox,
            "#ffffff" if is_stop_stage else color,
            linewidth=6.0 if is_stop_stage else 3.5,
            linestyle="-" if is_stop_stage else "--",
        )
        x1, y1, x2, y2 = crop_bbox
        ax_overview.text(
            x1,
            max(0, y1 - max(18, image.size[1] * 0.012)),
            f"S{stage['stage']}{' STOP' if is_stop_stage else ''}",
            color="#ffffff" if is_stop_stage else color,
            fontsize=14,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.68, "pad": 2.5, "edgecolor": "none"},
        )
    if stages:
        final_bbox = bbox_to_int_tuple(trace.get("final_bbox") or stages[-1]["crop_bbox"])
        draw_rect(ax_overview, final_bbox, "#ff2d55", linewidth=2.6, linestyle="-")
    ax_overview.set_title(
        "Original image overview: dashed boxes are stage crops; STOP is the selected final crop",
        fontsize=9.5,
    )
    ax_overview.set_xlim(0, image.size[0])
    ax_overview.set_ylim(image.size[1], 0)
    ax_overview.axis("off")

    for col in range(n_cols):
        ax_stage = fig.add_subplot(gs[2, col])
        if col >= len(stages):
            ax_stage.axis("off")
            continue

        stage = stages[col]
        crop_bbox = bbox_to_int_tuple(stage["crop_bbox"])
        crop_w = max(1, crop_bbox[2] - crop_bbox[0])
        crop_h = max(1, crop_bbox[3] - crop_bbox[1])
        crop_image = image.crop(crop_bbox)
        local_heatmap = stage.get("local_heatmap")
        if local_heatmap is None:
            ax_stage.imshow(crop_image)
        else:
            show_heatmap_overlay(ax_stage, crop_image, sanitize_heatmap(local_heatmap), alpha=0.42)
        is_stop_stage = (
            isinstance(stop_info, dict)
            and int(stop_info.get("selected_stage") or -1) == int(stage["stage"])
        )
        if is_stop_stage:
            draw_rect(ax_stage, (0, 0, crop_w, crop_h), "#ffffff", linewidth=4.0)
        ax_stage.set_title(
            f"Stage {stage['stage']}{' STOP' if is_stop_stage else ''} | {stage['target_label']}\n"
            + (
                f"crop {crop_w}x{crop_h} + local heatmap"
                if local_heatmap is not None
                else f"crop {crop_w}x{crop_h}; local heatmap skipped ({stage.get('stop_reason', 'terminal')})"
            ),
            fontsize=8.6,
        )
        ax_stage.set_xlim(0, crop_image.size[0])
        ax_stage.set_ylim(crop_image.size[1], 0)
        ax_stage.axis("off")
        # Keep the pre-heatmap object in the trace; the horizontal strip intentionally
        # shows one crop panel per stage to keep the stage flow compact.
        _ = local_heatmap

    if save:
        out_path = context.config.save_dir / f"{context.config.model}_{context.config.task}_{sample_id}_tile1024_initial_heatmap.png"
        fig.savefig(out_path, dpi=context.config.save_dpi, facecolor="white")
        print("saved:", out_path)
    plt.show()
    return fig


def heatmap_extent(heatmap: Any) -> tuple[int, int, int, int]:
    array = sanitize_heatmap(heatmap)
    return (0, int(array.shape[1]), int(array.shape[0]), 0)


def _mask_rgba(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    rgba_color = mcolors.to_rgba(color)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = rgba_color[0]
    rgba[..., 1] = rgba_color[1]
    rgba[..., 2] = rgba_color[2]
    rgba[..., 3] = mask.astype(np.float32) * float(alpha)
    return rgba


def _resize_mask_to_image(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = mask_image.resize(image_size, Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _centroid_heatmap_to_image(
    centroid_heatmap_xy: list[float] | tuple[float, float],
    heatmap_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    heatmap_height, heatmap_width = heatmap_shape
    image_width, image_height = image_size
    cx_heatmap, cy_heatmap = [float(v) for v in centroid_heatmap_xy]
    return (
        cx_heatmap / float(heatmap_width) * float(image_width),
        cy_heatmap / float(heatmap_height) * float(image_height),
    )


def _relative_bbox(bbox: Any, origin_bbox: Any) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    ox1, oy1, ox2, oy2 = bbox_to_int_tuple(origin_bbox)
    rx1 = max(x1, ox1) - ox1
    ry1 = max(y1, oy1) - oy1
    rx2 = min(x2, ox2) - ox1
    ry2 = min(y2, oy2) - oy1
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return (rx1, ry1, rx2, ry2)


def stage_hotspot_detail(trace: dict[str, Any], context: VizContext, stage_index: int = 1) -> dict[str, Any]:
    stages = trace["stages"]
    if not 1 <= int(stage_index) <= len(stages):
        raise IndexError(f"stage_index must be in [1, {len(stages)}], got {stage_index}")

    image = trace["image"]
    stage = stages[int(stage_index) - 1]
    heatmap = sanitize_heatmap(stage["pre_heatmap"])
    selected = stage["selection"].get("selected_component")
    if not selected:
        raise ValueError(f"stage {stage_index} has no selected hotspot component")

    mask_info = topmass_mask(heatmap, effective_topmass(context.config))
    topmass = mask_info["mask"]
    cluster_mask = np.zeros_like(topmass, dtype=bool)
    hx1, hy1, hx2, hy2 = bbox_to_int_tuple(selected["bbox"])
    hx1 = max(0, min(topmass.shape[1], hx1))
    hx2 = max(hx1, min(topmass.shape[1], hx2))
    hy1 = max(0, min(topmass.shape[0], hy1))
    hy2 = max(hy1, min(topmass.shape[0], hy2))
    cluster_mask[hy1:hy2, hx1:hx2] = topmass[hy1:hy2, hx1:hx2]

    crop_bbox = bbox_to_int_tuple(stage["crop_bbox"])
    crop_heatmap_window = image_bbox_to_heatmap_window(
        crop_bbox,
        image_size=image.size,
        heatmap_shape=heatmap.shape,
    )
    centroid_heatmap_xy = selected["centroid_heatmap_xy"]
    centroid_image_xy = _centroid_heatmap_to_image(
        centroid_heatmap_xy,
        heatmap_shape=heatmap.shape,
        image_size=image.size,
    )
    image_cluster_mask = _resize_mask_to_image(cluster_mask, image.size)

    return {
        "stage": stage,
        "heatmap": heatmap,
        "topmass_mask": topmass,
        "cluster_mask": cluster_mask,
        "image_cluster_mask": image_cluster_mask,
        "mask_info": {key: value for key, value in mask_info.items() if key != "mask"},
        "selected_component": selected,
        "component_heatmap_bbox": bbox_to_int_tuple(selected["bbox"]),
        "component_image_bbox": bbox_to_int_tuple(selected["image_bbox"]),
        "crop_bbox": crop_bbox,
        "crop_heatmap_window": crop_heatmap_window,
        "centroid_heatmap_xy": tuple(float(v) for v in centroid_heatmap_xy),
        "centroid_image_xy": centroid_image_xy,
    }


def _draw_centroid(ax: Any, xy: tuple[float, float], color: str = "#00ffff", label: str = "weighted centroid") -> None:
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=130,
        marker="x",
        linewidths=2.8,
        color=color,
        label=label,
        zorder=5,
    )


def _draw_detail_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.82)


def plot_hotspot_detail(
    trace: dict[str, Any],
    *,
    context: VizContext,
    stage_index: int = 1,
    save: bool = False,
    figsize: tuple[float, float] = (16, 12),
):
    detail = stage_hotspot_detail(trace, context, stage_index=stage_index)
    config = context.config
    image = trace["image"]
    row = trace["row"]
    sample_id = str(row.get("id", "unknown"))
    stage = detail["stage"]
    selected = detail["selected_component"]
    crop_bbox = detail["crop_bbox"]
    crop = image.crop(crop_bbox)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.82],
        left=0.04,
        right=0.985,
        bottom=0.04,
        top=0.90,
        hspace=0.23,
        wspace=0.08,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ],
        dtype=object,
    )
    zoom_ax = fig.add_subplot(gs[2, :])
    title = wrap_text(question_prompt_text(row), width=120, limit=430)
    fig.suptitle(
        f"{config.model} / {config.task} / sample {sample_id} / stage {stage_index} hotspot detail\n{title}",
        fontsize=11,
        fontweight="semibold",
        y=0.985,
    )

    ax = axes[0, 0]
    hm_extent = heatmap_extent(detail["heatmap"])
    ax.imshow(heatmap_for_display(detail["heatmap"]), cmap="inferno", interpolation="nearest", extent=hm_extent)
    ax.imshow(_mask_rgba(detail["topmass_mask"], "#ffd166", 0.18), interpolation="nearest", extent=hm_extent)
    ax.imshow(_mask_rgba(detail["cluster_mask"], "#ff2d55", 0.50), interpolation="nearest", extent=hm_extent)
    draw_heatmap_rect(ax, detail["component_heatmap_bbox"], "#ff2d55", label="selected hotspot cluster", linewidth=2.2)
    draw_heatmap_rect(ax, detail["crop_heatmap_window"], "#00d1ff", label="selected crop", linewidth=2.2, linestyle="--")
    _draw_centroid(ax, detail["centroid_heatmap_xy"])
    ax.set_title("Heatmap: cluster + weighted centroid", fontsize=10)
    ax.set_xlim(0, detail["heatmap"].shape[1])
    ax.set_ylim(detail["heatmap"].shape[0], 0)
    ax.axis("off")
    _draw_detail_legend(ax)

    ax = axes[0, 1]
    show_heatmap_overlay(ax, image, detail["heatmap"], alpha=0.42)
    ax.imshow(
        _mask_rgba(detail["image_cluster_mask"], "#ff2d55", 0.34),
        extent=(0, image.size[0], image.size[1], 0),
        interpolation="nearest",
    )
    draw_rect(ax, detail["component_image_bbox"], "#ff2d55", label="selected hotspot cluster", linewidth=2.5)
    draw_rect(ax, crop_bbox, "#00d1ff", label="selected crop", linewidth=2.5, linestyle="--")
    _draw_centroid(ax, detail["centroid_image_xy"])
    ax.set_title("Original image + heatmap overlay", fontsize=10)
    _draw_detail_legend(ax)

    ax = axes[1, 0]
    ax.imshow(image, extent=(0, image.size[0], image.size[1], 0))
    ax.imshow(
        _mask_rgba(detail["image_cluster_mask"], "#ff2d55", 0.42),
        extent=(0, image.size[0], image.size[1], 0),
        interpolation="nearest",
    )
    draw_rect(ax, detail["component_image_bbox"], "#ff2d55", label="selected hotspot cluster", linewidth=2.5)
    draw_rect(ax, crop_bbox, "#00d1ff", label="selected crop", linewidth=2.5, linestyle="--")
    _draw_centroid(ax, detail["centroid_image_xy"])
    ax.set_title("Original image: cluster + centroid + crop", fontsize=10)
    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)
    ax.axis("off")
    _draw_detail_legend(ax)

    ax = axes[1, 1]
    ax.axis("off")
    metrics = [
        f"Stage: {stage['stage']} ({stage['target_label']})",
        f"Ground truth: {trace['answer_text']}",
        f"Original score: {format_sample_score(trace.get('original_score'))}",
        f"New score: {format_sample_score(trace.get('crop_score'))}",
        f"Topmass: {effective_topmass(config):g}",
        f"Topmass threshold: {detail['mask_info']['threshold']:.6g}",
        f"Topmass covered fraction: {detail['mask_info']['covered_fraction']:.3f}",
        f"Cluster area: {selected['area']}",
        f"Cluster mass: {selected['mass']:.6g}",
        f"Cluster mean / peak: {selected['mean']:.6g} / {selected['peak']:.6g}",
        f"Cluster score: {selected['score']:.6g}",
        f"Centroid heatmap xy: ({detail['centroid_heatmap_xy'][0]:.2f}, {detail['centroid_heatmap_xy'][1]:.2f})",
        f"Centroid image xy: ({detail['centroid_image_xy'][0]:.1f}, {detail['centroid_image_xy'][1]:.1f})",
        f"Component image bbox: {detail['component_image_bbox']}",
        f"Crop bbox: {crop_bbox}",
        f"Original image: {trace.get('image_path', 'not available')}",
    ]
    ax.text(0.0, 1.0, "\n".join(metrics), va="top", ha="left", fontsize=10, linespacing=1.35)
    rel_component = _relative_bbox(detail["component_image_bbox"], crop_bbox)
    zoom_ax.axis("off")
    if rel_component is not None:
        zoom_ax.imshow(crop, extent=(0, crop.size[0], crop.size[1], 0))
        draw_rect(zoom_ax, rel_component, "#ff2d55", linewidth=2.0)
        cx, cy = detail["centroid_image_xy"]
        zoom_ax.scatter([cx - crop_bbox[0]], [cy - crop_bbox[1]], s=110, marker="x", linewidths=2.5, color="#00ffff")
        zoom_ax.set_title("Selected crop zoom (no heatmap)", fontsize=10)
        zoom_ax.set_xlim(0, crop.size[0])
        zoom_ax.set_ylim(crop.size[1], 0)
    else:
        zoom_ax.text(0.5, 0.5, "Selected crop zoom not available", ha="center", va="center", fontsize=10)

    if save:
        out_path = config.save_dir / f"{config.model}_{config.task}_{sample_id}_stage{stage_index}_hotspot_detail.png"
        fig.savefig(out_path, dpi=config.save_dpi, facecolor="white")
        print("saved:", out_path)
    plt.show()
    return fig


def plot_all_hotspot_details(
    trace: dict[str, Any],
    *,
    context: VizContext,
    stage_indices: list[int] | tuple[int, ...] | None = None,
    save: bool = False,
    figsize: tuple[float, float] = (16, 12),
) -> list[Any]:
    if stage_indices is None:
        stage_indices = tuple(range(1, len(trace["stages"]) + 1))
    figs = []
    for stage_index in stage_indices:
        print(f"plotting hotspot detail for stage {stage_index}/{len(trace['stages'])}")
        figs.append(
            plot_hotspot_detail(
                trace,
                context=context,
                stage_index=int(stage_index),
                save=save,
                figsize=figsize,
            )
        )
    return figs


def question_title(text: Any) -> str:
    for line in str(text).splitlines():
        clean = line.strip()
        if clean:
            return clean
    return ""


def question_prompt_text(row: dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    base = normalize_answer_text(row.get("short_question")) or question_title(question)
    choices = mcq_choices_for_row(row)
    if choices:
        choice_text = " | ".join(f"{key}. {choices[key]}" for key in sorted(choices))
        return f"{base}  {choice_text}"
    return base


def wrap_text(text: Any, width: int = 96, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if limit is not None and len(text) > limit:
        text = text[: limit - 3] + "..."
    return "\n".join(
        textwrap.wrap(
            text,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def info_line(label: str, value: Any, width: int = 31) -> str:
    text = display_answer(value, 130)
    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["not available"]
    indent = " " * (len(label) + 2)
    return "\n".join([f"{label}: {wrapped[0]}"] + [f"{indent}{line}" for line in wrapped[1:]])


def plot_trace(trace: dict[str, Any], *, context: VizContext, save: bool = True):
    config = context.config
    image = trace["image"]
    stages = trace["stages"]
    row = trace["row"]
    sample_id = str(row.get("id", "unknown"))
    n_stages = max(1, len(stages))

    fig = plt.figure(figsize=config.ppt_figsize, facecolor="white")
    title = wrap_text(question_prompt_text(row), width=118, limit=430)
    fig.suptitle(title, fontsize=10.8, y=0.975, fontweight="semibold", ha="center")

    ax_text = fig.add_axes([0.045, 0.115, 0.225, 0.635])
    ax_text.axis("off")
    original_source = trace.get("original_answer_source", "missing")
    crop_source = trace.get("crop_answer_source", "missing")
    text_block = "\n\n".join(
        [
            info_line("Ground truth", trace["answer_text"]),
            info_line("Original answer", f"{display_answer(trace.get('original_answer'), 130)} [{original_source}]"),
            info_line("Original score", format_sample_score(trace.get("original_score"))),
            info_line("New answer", f"{display_answer(trace.get('crop_answer'), 130)} [{crop_source}]"),
            info_line("New score", format_sample_score(trace.get("crop_score"))),
            f"Model: {config.model}",
            f"Dataset: {config.task}",
            f"Sample: {sample_id}",
            "Blue box = selected crop",
            (
                "Final input = red-box original + ViT floor + question"
                if trace.get("final_input_mode") == FINAL_INPUT_REDBOX_VITFLOOR_QUESTION
                else "Red box = final top-1 cluster"
                if trace.get("final_input_mode") == FINAL_INPUT_BOXED_ORIGINAL
                else ""
            ),
        ]
    )
    ax_text.text(0.0, 1.0, text_block, va="top", ha="left", fontsize=9.2, linespacing=1.18)

    gs = fig.add_gridspec(
        2,
        n_stages,
        left=0.305,
        right=0.975,
        bottom=0.085,
        top=0.755,
        width_ratios=[1.0] * n_stages,
        height_ratios=[1.0, 1.0],
        wspace=0.08,
        hspace=0.20,
    )

    for col in range(n_stages):
        stage = stages[col] if col < len(stages) else None
        if stage is None:
            crop_bbox = bbox_to_int_tuple(
                trace.get("final_bbox") or (0, 0, image.size[0], image.size[1])
            )
            crop = image.crop(crop_bbox)
            selected_component = None
            pre_heatmap = trace.get("final_heatmap", trace.get("initial_heatmap"))
            stage_title = "Final heatmap\nmainline final crop"
        else:
            crop_bbox = bbox_to_int_tuple(stage["crop_bbox"])
            crop = image.crop(crop_bbox)
            selected_component = stage["selection"].get("selected_component")
            if selected_component:
                comp_text = f"area={selected_component['area']} score={stage['selection']['score']:.2g}"
            else:
                comp_text = "fallback"
            pre_heatmap = stage["pre_heatmap"]
            stage_title = f"Stage {stage['stage']} | {stage['target_label']}\nselect crop ({comp_text})"

        ax = fig.add_subplot(gs[0, col])
        show_heatmap_overlay(ax, image, pre_heatmap, alpha=0.45)
        if stage is not None:
            draw_rect(ax, crop_bbox, "#00d1ff", linewidth=2.6)
        ax.set_title(stage_title, fontsize=8.5)

        ax = fig.add_subplot(gs[1, col])
        if col == n_stages - 1:
            if trace.get("final_input_mode") in {
                FINAL_INPUT_BOXED_ORIGINAL,
                FINAL_INPUT_REDBOX_VITFLOOR_QUESTION,
            }:
                final_input_image = trace["final_input_image"]
                ax.imshow(
                    final_input_image,
                    extent=(0, final_input_image.size[0], final_input_image.size[1], 0),
                )
                ax.set_xlim(0, final_input_image.size[0])
                ax.set_ylim(final_input_image.size[1], 0)
                if trace.get("final_input_mode") == FINAL_INPUT_REDBOX_VITFLOOR_QUESTION:
                    title = "Final input: red-box original + ViT floor\nText prompt: original question"
                else:
                    title = "Final input: original + red top-1 cluster box"
            else:
                final_crop = trace["final_input_image"]
                ax.imshow(final_crop, extent=(0, final_crop.size[0], final_crop.size[1], 0))
                ax.set_xlim(0, final_crop.size[0])
                ax.set_ylim(final_crop.size[1], 0)
                title = f"Final crop image (no heatmap)\n{final_crop.size[0]}x{final_crop.size[1]}"
            ax.axis("off")
            ax.set_title(title, fontsize=8.5)
        else:
            if stage is None:
                ax.imshow(crop, extent=(0, crop.size[0], crop.size[1], 0))
                ax.set_title(f"Final square crop\n{crop.size[0]}x{crop.size[1]}", fontsize=8.5)
            else:
                local_heatmap = stage.get("local_heatmap")
                if local_heatmap is None:
                    ax.imshow(crop, extent=(0, crop.size[0], crop.size[1], 0))
                    ax.set_title(
                        f"Crop; local heatmap skipped\n{crop.size[0]}x{crop.size[1]}",
                        fontsize=8.5,
                    )
                else:
                    show_heatmap_overlay(ax, crop, local_heatmap, alpha=0.50)
                    ax.set_title(f"Crop heatmap\n{crop.size[0]}x{crop.size[1]}", fontsize=8.5)

    if save:
        out_path = config.save_dir / f"{config.model}_{config.task}_{sample_id}_top1_hotspot_multistage_ppt.png"
        fig.savefig(out_path, dpi=config.save_dpi, facecolor="white")
        print("saved:", out_path)
    plt.show()
    return fig


def visualize_one(
    *,
    context: VizContext,
    adapter: Any,
    sample_index: int | None = None,
    sample_id: str | None = None,
    save: bool = True,
    show_image_link: bool = True,
) -> tuple[dict[str, Any], Any]:
    config = context.config
    sample_index = config.sample_index if sample_index is None else sample_index
    sample_id = config.sample_id if sample_id is None else sample_id
    row = choose_sample(context.rows, sample_id=sample_id, sample_index=int(sample_index))
    print("running sample index/id =", sample_index, row.get("id"))
    print("original image =", original_image_path(context, row))
    if show_image_link:
        display_original_image_link(context, row)
    trace = run_visual_trace(row, context=context, adapter=adapter)
    fig = plot_trace(trace, context=context, save=save)
    return trace, fig
