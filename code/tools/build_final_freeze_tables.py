#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import numpy as np
from PIL import Image
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "code/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import get_score  # noqa: E402
from info import task_to_image_path  # noqa: E402


get_score.tqdm = lambda iterable, **_: iterable

FINAL_ROOT = ROOT / "outputs/model_outputs/final_freeze_20260713"
BLIP_MAINLINE = ROOT / "outputs/model_outputs/mainline/blip_subtraction_tile1024_stop1"
ABLATION_ROOT = ROOT / "outputs/model_outputs/ablations"
BASE_STD = ROOT / "outputs/model_outputs/baselines/original_methods_nonhr_3model_7task"
BASE_HR = ROOT / "outputs/model_outputs/baselines/original_methods_highres_vstar_hrbench_3model"
BASE_HR_LOW = ROOT / "outputs/model_outputs/baselines/original_methods_hrbench_lowres_3model_2task"
QWEN_RETAINED_HR8 = (
    ROOT / "outputs/model_outputs/analysis/qwen_topmass_hrbench_20260714"
    / "qwen2_5/hrbench_8k/tm0p55"
)
TEXTVQA_FULL = ROOT / "data/textvqa/data.json"
RESOLUTION = ROOT / "results/tables/analysis/dataset_resolution_inventory.csv"
OUT = ROOT / "results/tables/final_freeze_20260713"
SOURCE_MANIFEST = OUT / "final_result_source_manifest.json"
MODELS = ("llava", "blip", "qwen2_5")
ABLATION_MODELS = ("llava", "blip")
TASKS = (
    "aokvqa", "docvqa", "gqa", "pope", "textvqa", "vqav2",
    "vstar", "hrbench_4k", "hrbench_8k",
)
CORE_TASKS = ("vstar", "hrbench_4k", "hrbench_8k")
ABLATION_TASKS = TASKS
METHODS = ("rel_att", "grad_att", "pure_grad")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_in(directory: Path, model: str, task: str) -> Path:
    manifest = directory / "run_manifest.json"
    if manifest.is_file():
        output = Path(str(load_json(manifest).get("output", "")))
        if output.is_file():
            return output
    matches = [
        path for path in directory.glob(f"{model}-{task}-*.json")
        if path.name not in {"evaluation_report.json", "run_manifest.json"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one prediction under {directory}, got {matches}")
    return matches[0]


@lru_cache(maxsize=1)
def packaged_source_index() -> dict[tuple[str, str, str, str], Path]:
    """Resolve the curated package's selected, dereferenced result entries.

    The source package deliberately keeps result files as symlinks so a local
    result archive is not duplicated.  Therefore this resolver uses the
    package manifest rather than the historical absolute output paths recorded
    in individual run manifests.
    """
    manifest = load_json(SOURCE_MANIFEST)
    index: dict[tuple[str, str, str, str], Path] = {}
    for item in manifest["sources"]:
        key = (str(item["family"]), str(item["model"]), str(item["task"]), str(item["method"]))
        path = Path(str(item["path"]))
        if path.is_absolute():
            raise RuntimeError(f"packaged source manifest must use relative paths: {path}")
        if key in index:
            raise RuntimeError(f"duplicate packaged source: {key}")
        index[key] = ROOT / path
    return index


def packaged_prediction(family: str, model: str, task: str, method: str) -> Path:
    key = (family, model, task, method)
    try:
        path = packaged_source_index()[key]
    except KeyError as exc:
        raise KeyError(f"missing packaged prediction: {key}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def final_prediction(model: str, task: str) -> Path:
    if model == "qwen2_5":
        return packaged_prediction("final_phf", model, task, "phf_crop")
    return packaged_prediction("core_ablation", model, task, "full_phf")


def ablation_prediction(variant: str, model: str, task: str) -> Path:
    if variant not in {"full_phf", "iterative_local", "first_stage_only"}:
        raise KeyError(variant)
    return packaged_prediction("core_ablation", model, task, variant)


def textvqa_full_ids() -> set[str]:
    ids = {str(row["id"]) for row in load_json(TEXTVQA_FULL)}
    if len(ids) != 5000:
        raise RuntimeError(f"expected 5,000 unique TextVQA IDs, got {len(ids)}")
    return ids


TEXTVQA_FULL_IDS = textvqa_full_ids()


def filtered_rows(path: Path, task: str) -> list[dict[str, Any]]:
    rows = load_json(path)
    if task == "textvqa":
        row_ids = {str(row["id"]) for row in rows}
        if len(rows) != 5000 or row_ids != TEXTVQA_FULL_IDS:
            raise RuntimeError(f"TextVQA5000 ID mismatch for {path}: rows={len(rows)}, unique={len(row_ids)}")
    return rows


def evaluate_rows(rows: list[dict[str, Any]], task: str) -> tuple[float, float]:
    if task == "docvqa":
        def anls(key: str) -> float:
            return 100.0 * fmean(get_score.get_anls(str(row[key]), list(row["labels"])) for row in rows)
        return anls("original_answer"), anls("crop_answer")
    fn: Callable[[list[dict[str, Any]]], tuple[float, float]] = {
        "textvqa": get_score.evaluate_textvqa,
        "vstar": get_score.evaluate_vstar,
        "hrbench_4k": get_score.evaluate_vstar,
        "hrbench_8k": get_score.evaluate_vstar,
        "pope": get_score.evaluate_pope,
        "aokvqa": get_score.evaluate_aokvqa,
        "vqav2": get_score.evaluate_vqav2,
        "gqa": get_score.evaluate_gqa,
    }[task]
    return tuple(float(value) for value in fn(rows))


def score_file(path: Path, task: str) -> tuple[float, float, int]:
    rows = filtered_rows(path, task)
    raw, crop = evaluate_rows(rows, task)
    return raw, crop, len(rows)


def standard_baseline_path(model: str, task: str, method: str) -> Path:
    return packaged_prediction("vicrop_baseline", model, task, method)


def high_baseline_path(model: str, task: str, method: str) -> Path | None:
    # The package manifest already records the stronger standard/high-resolution
    # result selected for every model/task/method cell.
    return None


def labels() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"llava": "LLaVA", "blip": "InstructBLIP", "qwen2_5": "Qwen2.5-VL"},
        {
            "aokvqa": "AOKVQA", "docvqa": "DocVQA", "gqa": "GQA", "pope": "POPE",
            "textvqa": "TextVQA", "vqav2": "VQAv2", "vstar": "VSTAR",
            "hrbench_4k": "HRBench-4K", "hrbench_8k": "HRBench-8K",
        },
    )


MODEL_LABEL, TASK_LABEL = labels()


def resolution_groups() -> dict[str, str]:
    with RESOLUTION.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["task"]: ("High" if row["resolution_group"].startswith("High") else "Standard")
            for row in csv.DictReader(handle)
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" if column in {"Resolution", "Model", "Task", "Comparison"} else "---:" for column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main_table(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = resolution_groups()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for task in TASKS:
            phf_path = final_prediction(model, task)
            raw, phf, count = score_file(phf_path, task)
            method_scores: dict[str, float] = {}
            method_sources: dict[str, str] = {}
            for method in METHODS:
                candidates: list[tuple[float, Path]] = []
                standard = standard_baseline_path(model, task, method)
                candidates.append((score_file(standard, task)[1], standard))
                high = high_baseline_path(model, task, method)
                if high is not None:
                    candidates.append((score_file(high, task)[1], high))
                value, source = max(candidates, key=lambda item: item[0])
                method_scores[method] = value
                method_sources[method] = str(source.relative_to(ROOT))
                sources.append({
                    "family": "vicrop_baseline", "model": model, "task": task,
                    "method": method, "path": str(source), "count": score_file(source, task)[2],
                })
            best = max(method_scores.values())
            rows.append({
                "Resolution": groups[task], "Model": MODEL_LABEL[model], "Task": TASK_LABEL[task], "N": count,
                "NoCrop": f"{raw:.2f}", "Rel-Att": f"{method_scores['rel_att']:.2f}",
                "Grad-Att": f"{method_scores['grad_att']:.2f}", "Pure-Grad": f"{method_scores['pure_grad']:.2f}",
                "PHF-Crop": f"{phf:.2f}", "Gain vs best ViCrop": f"{phf - best:+.2f}",
            })
            sources.append({
                "family": "final_phf", "model": model, "task": task, "method": "phf_crop",
                "path": str(phf_path.relative_to(ROOT)), "count": count,
            })
    return rows


def core_ablation(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ABLATION_MODELS:
        for task in ABLATION_TASKS:
            paths = {
                variant: ablation_prediction(variant, model, task)
                for variant in ("first_stage_only", "iterative_local", "full_phf")
            }
            full_rows = filtered_rows(paths["full_phf"], task)
            nocrop, full = evaluate_rows(full_rows, task)
            first = score_file(paths["first_stage_only"], task)[1]
            iterative = score_file(paths["iterative_local"], task)[1]
            rows.append({
                "Model": MODEL_LABEL[model], "Task": TASK_LABEL[task], "N": len(full_rows),
                "NoCrop": f"{nocrop:.2f}", "First-stage only": f"{first:.2f}",
                "Iterative Local": f"{iterative:.2f}", "Full PHF": f"{full:.2f}",
                "Fusion gain": f"{full - iterative:+.2f}",
            })
            for variant, path in paths.items():
                sources.append({
                    "family": "core_ablation", "model": model, "task": task,
                    "method": variant, "path": str(path.relative_to(ROOT)), "count": len(filtered_rows(path, task)),
                })
    return rows


def image_path(task: str, row: dict[str, Any]) -> Path:
    path = Path(str(row["image_path"]))
    return path if path.is_absolute() else Path(task_to_image_path[task]) / path


def image_max_side(task: str, row: dict[str, Any]) -> int:
    with Image.open(image_path(task, row)) as image:
        return max(int(image.width), int(image.height))


def initial_forwards(row: dict[str, Any], task: str) -> int:
    info = row.get("qpes_initial_heatmap_tile_mosaic") or {}
    if info.get("tile_count") is not None:
        return int(info["tile_count"])
    side = image_max_side(task, row)
    if side <= 1024:
        return 1
    with Image.open(image_path(task, row)) as image:
        return int(math.ceil(image.width / 1024) * math.ceil(image.height / 1024))


def trace_summary(rows: list[dict[str, Any]], task: str) -> dict[str, float]:
    totals, local, stages, stopped = [], [], [], []
    for row in rows:
        trace = list(row.get("qpes_multistage_trace") or [])
        local_count = sum(stage.get("local_heatmap_shape") is not None for stage in trace)
        init = initial_forwards(row, task)
        totals.append(init + local_count)
        local.append(local_count)
        stages.append(len(trace))
        stop = row.get("qpes_stop_reason") == "hotspot_fills_candidate" or bool(
            (row.get("qpes_adaptive_stop") or {}).get("stopped_by_rule")
        )
        stopped.append(stop)
    return {
        "Avg initial HM": fmean(initial_forwards(row, task) for row in rows),
        "Avg local HM": fmean(local), "Avg total HM": fmean(totals),
        "Avg stages": fmean(stages), "Stop rate (%)": 100.0 * fmean(stopped),
    }


def efficiency_table() -> list[dict[str, Any]]:
    rows = []
    for model in ABLATION_MODELS:
        for task in ABLATION_TASKS:
            for variant, label in (("iterative_local", "Iterative Local"), ("full_phf", "Full PHF")):
                data = filtered_rows(ablation_prediction(variant, model, task), task)
                summary = trace_summary(data, task)
                rows.append({
                    "Model": MODEL_LABEL[model], "Task": TASK_LABEL[task], "Method": label,
                    "N": len(data), **{key: f"{value:.3f}" for key, value in summary.items()},
                })
    return rows


def mcq_correct(row: dict[str, Any], key: str) -> bool:
    answer = str(row[key])
    letter = get_score.normalize_mcq_letter(answer)
    if not letter:
        candidates = str(row["question"]).split("\n")[1:-1]
        letter = "ABCD"[get_score.get_index(answer, candidates)]
    return letter == str(row["labels"]).strip().upper()


def resolution_bins() -> list[dict[str, Any]]:
    bins = ((0, 1024, "<=1024"), (1025, 2048, "1025-2048"), (2049, 4096, "2049-4096"), (4097, 10**9, ">4096"))
    rows = []
    for model in ABLATION_MODELS:
        for task in CORE_TASKS:
            full = filtered_rows(ablation_prediction("full_phf", model, task), task)
            iterative_by_id = {str(row["id"]): row for row in filtered_rows(ablation_prediction("iterative_local", model, task), task)}
            grouped: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
            for row in full:
                side = image_max_side(task, row)
                label = next(name for low, high, name in bins if low <= side <= high)
                other = iterative_by_id[str(row["id"])]
                grouped[label].append((mcq_correct(row, "original_answer"), mcq_correct(other, "crop_answer"), mcq_correct(row, "crop_answer")))
            for _low, _high, label in bins:
                values = grouped.get(label, [])
                if not values:
                    continue
                n = len(values)
                nocrop = 100.0 * sum(value[0] for value in values) / n
                iterative = 100.0 * sum(value[1] for value in values) / n
                full_score = 100.0 * sum(value[2] for value in values) / n
                rows.append({
                    "Model": MODEL_LABEL[model], "Task": TASK_LABEL[task], "Max side": label, "N": n,
                    "NoCrop": f"{nocrop:.2f}", "Iterative Local": f"{iterative:.2f}",
                    "Full PHF": f"{full_score:.2f}", "PHF-NoCrop": f"{full_score - nocrop:+.2f}",
                    "PHF-Iterative": f"{full_score - iterative:+.2f}",
                })
    return rows


def significance_table() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260713)
    rows = []
    for model in ABLATION_MODELS:
        for task in CORE_TASKS:
            full_rows = filtered_rows(ablation_prediction("full_phf", model, task), task)
            full_by_id = {str(row["id"]): row for row in full_rows}
            comparators = {
                "NoCrop": (full_by_id, "original_answer"),
                "First-stage only": ({str(row["id"]): row for row in filtered_rows(ablation_prediction("first_stage_only", model, task), task)}, "crop_answer"),
                "Iterative Local": ({str(row["id"]): row for row in filtered_rows(ablation_prediction("iterative_local", model, task), task)}, "crop_answer"),
            }
            ids = sorted(full_by_id)
            method = np.asarray([100.0 * mcq_correct(full_by_id[item], "crop_answer") for item in ids])
            for name, (baseline_by_id, key) in comparators.items():
                if baseline_by_id.keys() != full_by_id.keys():
                    raise RuntimeError(f"paired id mismatch: {model}/{task}/{name}")
                baseline = np.asarray([100.0 * mcq_correct(baseline_by_id[item], key) for item in ids])
                differences = method - baseline
                samples = differences[rng.integers(0, len(ids), size=(3000, len(ids)))].mean(axis=1)
                low, high = np.percentile(samples, [2.5, 97.5])
                method_only = int(np.sum((method == 100.0) & (baseline == 0.0)))
                baseline_only = int(np.sum((method == 0.0) & (baseline == 100.0)))
                discordant = method_only + baseline_only
                p = 1.0 if discordant == 0 else float(binomtest(method_only, discordant, 0.5).pvalue)
                rows.append({
                    "Model": MODEL_LABEL[model], "Task": TASK_LABEL[task], "Comparison": f"Full PHF vs {name}",
                    "N": len(ids), "Delta": f"{differences.mean():+.2f}", "CI95 low": f"{low:+.2f}",
                    "CI95 high": f"{high:+.2f}", "p (McNemar)": f"{p:.6g}",
                    "PHF-only correct": method_only, "Comparator-only correct": baseline_only,
                })
    return rows


def docvqa_anls_table() -> list[dict[str, Any]]:
    def anls(rows: list[dict[str, Any]], key: str) -> float:
        return 100.0 * fmean(
            get_score.get_anls(str(row[key]), list(row["labels"])) for row in rows
        )

    rows = []
    for model in MODELS:
        phf_rows = filtered_rows(final_prediction(model, "docvqa"), "docvqa")
        values = {
            "NoCrop": anls(phf_rows, "original_answer"),
            "PHF-Crop": anls(phf_rows, "crop_answer"),
        }
        for method in METHODS:
            baseline_rows = filtered_rows(
                standard_baseline_path(model, "docvqa", method), "docvqa"
            )
            values[method] = anls(baseline_rows, "crop_answer")
        rows.append({
            "Model": MODEL_LABEL[model], "N": len(phf_rows),
            "NoCrop": f"{values['NoCrop']:.2f}",
            "Rel-Att": f"{values['rel_att']:.2f}",
            "Grad-Att": f"{values['grad_att']:.2f}",
            "Pure-Grad": f"{values['pure_grad']:.2f}",
            "PHF-Crop": f"{values['PHF-Crop']:.2f}",
        })
    return rows


def write_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    main_rows = main_table(sources)
    ablation_rows = core_ablation(sources)
    efficiency_rows = efficiency_table()
    resolution_rows = resolution_bins()
    significance_rows = significance_table()
    docvqa_rows = docvqa_anls_table()
    artifacts = {
        "main_table": (main_rows, ["Resolution", "Model", "Task", "N", "NoCrop", "Rel-Att", "Grad-Att", "Pure-Grad", "PHF-Crop", "Gain vs best ViCrop"]),
        "core_ablation": (ablation_rows, ["Model", "Task", "N", "NoCrop", "First-stage only", "Iterative Local", "Full PHF", "Fusion gain"]),
        "efficiency": (efficiency_rows, list(efficiency_rows[0])),
        "resolution_bins": (resolution_rows, list(resolution_rows[0])),
        "paired_significance": (significance_rows, list(significance_rows[0])),
        "docvqa_anls": (docvqa_rows, list(docvqa_rows[0])),
    }
    for name, (rows, columns) in artifacts.items():
        write_csv(OUT / f"{name}.csv", rows)
        title = name.replace("_", " ").title()
        note = ""
        if name == "main_table":
            note = (
                "\nFor each ViCrop variant, the stronger standard/high-resolution implementation is reported. "
                "TextVQA merges the frozen 500-example calibration partition with the disjoint 4,500-example held-out partition and reports all 5,000 examples. DocVQA uses ANLS. Scores are recomputed from prediction JSONs.\n"
            )
        (OUT / f"{name}.md").write_text(
            f"# {title}\n{note}\n" + md_table(columns, rows) + "\n", encoding="utf-8"
        )
    paper_ablation = ROOT / "results/tables/paper/core_ablations.md"
    paper_ablation.write_text(
        "# Core Ablations\n\n"
        "LLaVA and InstructBLIP are evaluated on all nine retained datasets. "
        "Bootstrap, topmass, crop schedule, stopping rule, answer policy, and scorer are fixed within each model/task comparison.\n\n"
        + md_table(
            ["Model", "Task", "N", "NoCrop", "First-stage only", "Iterative Local", "Full PHF", "Fusion gain"],
            ablation_rows,
        )
        + "\n\nDetailed definitions and interpretation boundaries: "
        "[`../../../docs/ablation_experiments.md`](../../../docs/ablation_experiments.md).\n",
        encoding="utf-8",
    )
    unique_sources = sorted(
        {str(item["path"]): item for item in sources}.values(),
        key=lambda item: (item["family"], item["model"], item["task"], item["method"]),
    )
    manifest = {
        "version": "final_freeze_20260713",
        "textvqa_protocol": "500 calibration + disjoint 4500 held-out, merged by original TextVQA ID",
        "source_count": len(unique_sources),
        "sources": unique_sources,
        "tables": [str((OUT / f"{name}.csv").relative_to(ROOT)) for name in artifacts],
    }
    (OUT / "final_result_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    write_outputs()
