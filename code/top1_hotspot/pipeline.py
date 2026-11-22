from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PIL import Image

from .components import hotspot_component_square_bbox
from .adaptive_stop import component_short_ratio
from .geometry import (
    bbox_to_int_tuple,
    image_bbox_to_heatmap_window,
    jsonable,
    normalize_heatmap_sum,
    resize_heatmap,
    sanitize_heatmap,
    target_label,
)
from .heatmap_stats import compute_normalized_heatmap_entropy


FUSION_MODE = "coverage_average"
LOCALIZATION_STRATEGIES = ("phf", "iterative_local")


def add_sum_normalized_contribution_average(
    sum_map: np.ndarray,
    count_map: np.ndarray,
    local_heatmap: Any,
    bbox: Any,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    window = image_bbox_to_heatmap_window(
        bbox,
        image_size=image_size,
        heatmap_shape=sum_map.shape,
    )
    hx1, hy1, hx2, hy2 = window
    local_resized = resize_heatmap(local_heatmap, (hy2 - hy1, hx2 - hx1))
    local_resized = normalize_heatmap_sum(local_resized)
    sum_map[hy1:hy2, hx1:hx2] += local_resized
    count_map[hy1:hy2, hx1:hx2] += 1.0
    fused = sum_map / np.maximum(count_map, 1e-6)
    return sanitize_heatmap(fused), window


def compact_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return jsonable(
        {
            key: value
            for key, value in stage.items()
            if key not in {"pre_heatmap", "local_heatmap", "fused_heatmap"}
        }
    )


def run_top1_hotspot_chain(
    *,
    image: Image.Image,
    adapter: Any,
    attention_prompt: str,
    initial_heatmap: Any,
    targets: list[dict[str, Any]],
    topmass: float,
    adaptive_stop_hotspot_ratio: float = 1.0,
    local_heatmap_fn: Callable[[Image.Image], tuple[Any, dict[str, Any] | None]] | None = None,
    stage_candidate_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    import torch

    current_heatmap = sanitize_heatmap(initial_heatmap)
    heatmap_sum = normalize_heatmap_sum(current_heatmap).copy()
    heatmap_count = np.ones_like(heatmap_sum, dtype=np.float32)
    stages: list[dict[str, Any]] = []

    for stage_index, target in enumerate(targets, start=1):
        is_terminal_stage = stage_index == len(targets)
        side = int(target["side"])
        selection = (
            stage_candidate_scorer(current_heatmap, image.size, side, topmass=topmass)
            if stage_candidate_scorer is not None
            else hotspot_component_square_bbox(
                current_heatmap,
                image.size,
                side,
                topmass=topmass,
            )
        )
        crop_bbox = bbox_to_int_tuple(selection["bbox"])
        selected_component = selection.get("selected_component")
        component_bbox = (
            selected_component.get("image_bbox")
            if isinstance(selected_component, dict)
            else None
        )
        stage = {
            "stage": int(stage_index),
            "target_kind": target["kind"],
            "target_ratio": target["ratio"],
            "target_label": target_label(target, adapter.min_crop_side),
            "side": int(side),
            "selection": selection,
            "component_image_bbox": component_bbox,
            "crop_bbox": crop_bbox,
            "pre_heatmap": current_heatmap,
            "pre_entropy": float(compute_normalized_heatmap_entropy(current_heatmap)),
        }
        stop_ratio = component_short_ratio(stage)
        stage["adaptive_stop_component_short_ratio"] = float(stop_ratio)
        if isinstance(selected_component, dict) and adaptive_stop_hotspot_ratio is not None:
            if stop_ratio >= float(adaptive_stop_hotspot_ratio):
                stage["adaptive_stop_triggered"] = True
                stage["adaptive_stop_stopped_before_local_heatmap"] = True
                stage["stop_reason"] = "hotspot_fills_candidate"
                stage["final_bbox_semantics"] = selection.get(
                    "bbox_semantics", "current_centered_square_crop"
                )
                stages.append(stage)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

        if is_terminal_stage:
            stage["terminal_crop"] = True
            stage["local_heatmap_skipped"] = "no_later_stage_uses_it"
            stage["stop_reason"] = "min_crop_side"
            stage["final_bbox_semantics"] = selection.get(
                "bbox_semantics", "current_centered_square_crop"
            )
            stages.append(stage)
            break

        crop = image.crop(crop_bbox)
        with torch.no_grad():
            if local_heatmap_fn is None:
                local_heatmap = adapter.heatmap_fn(
                    crop,
                    attention_prompt,
                    adapter.general_prompt,
                    adapter.model,
                    adapter.processor,
                )
                local_heatmap_info = None
            else:
                local_heatmap, local_heatmap_info = local_heatmap_fn(crop)
        local_heatmap = sanitize_heatmap(local_heatmap)
        fused_heatmap, fused_window = add_sum_normalized_contribution_average(
            heatmap_sum,
            heatmap_count,
            local_heatmap,
            crop_bbox,
            image.size,
        )
        stage.update(
            {
                "local_heatmap": local_heatmap,
                "local_heatmap_shape": [int(v) for v in local_heatmap.shape],
                "local_heatmap_tile_mosaic": jsonable(local_heatmap_info),
                "fused_heatmap": fused_heatmap,
                "fused_heatmap_window": [int(v) for v in fused_window],
                "fused_entropy": float(compute_normalized_heatmap_entropy(fused_heatmap)),
                "fusion_count_max": float(heatmap_count.max()),
                "fusion_mode": FUSION_MODE,
            }
        )
        stages.append(stage)
        current_heatmap = fused_heatmap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return stages


def _offset_bbox(
    bbox: Any,
    offset_xy: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    offset_x, offset_y = offset_xy
    return (
        int(x1 + offset_x),
        int(y1 + offset_y),
        int(x2 + offset_x),
        int(y2 + offset_y),
    )


def _selection_in_original_coordinates(
    selection: dict[str, Any],
    offset_xy: tuple[int, int],
) -> dict[str, Any]:
    translated = dict(selection)
    translated["bbox"] = _offset_bbox(selection["bbox"], offset_xy)
    for key in ("selected_component", "runner_up_component"):
        component = selection.get(key)
        if not isinstance(component, dict):
            translated[key] = None
            continue
        translated_component = dict(component)
        if component.get("image_bbox") is not None:
            translated_component["image_bbox"] = _offset_bbox(component["image_bbox"], offset_xy)
        translated[key] = translated_component
    return translated


def run_iterative_local_chain(
    *,
    image: Image.Image,
    adapter: Any,
    attention_prompt: str,
    initial_heatmap: Any,
    targets: list[dict[str, Any]],
    topmass: float,
    adaptive_stop_hotspot_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    """Run the same crop schedule without retaining or fusing past heatmaps.

    Each stage selects the next crop only from the heatmap of the current image
    frame. The selected local box is translated back to original-image
    coordinates for final answering and trace comparability.
    """
    import torch

    frame_image = image
    frame_bbox = (0, 0, int(image.width), int(image.height))
    current_heatmap = sanitize_heatmap(initial_heatmap)
    stages: list[dict[str, Any]] = []

    for stage_index, target in enumerate(targets, start=1):
        is_terminal_stage = stage_index == len(targets)
        target_side = int(target["side"])
        side = min(target_side, frame_image.width, frame_image.height)
        local_selection = hotspot_component_square_bbox(
            current_heatmap,
            frame_image.size,
            side,
            topmass=topmass,
        )
        offset_xy = (int(frame_bbox[0]), int(frame_bbox[1]))
        selection = _selection_in_original_coordinates(local_selection, offset_xy)
        crop_bbox = bbox_to_int_tuple(selection["bbox"])
        selected_component = selection.get("selected_component")
        component_bbox = (
            selected_component.get("image_bbox")
            if isinstance(selected_component, dict)
            else None
        )
        stage = {
            "stage": int(stage_index),
            "target_kind": target["kind"],
            "target_ratio": target["ratio"],
            "target_label": target_label(target, adapter.min_crop_side),
            "side": int(side),
            "selection": selection,
            "component_image_bbox": component_bbox,
            "crop_bbox": crop_bbox,
            "search_frame_bbox": bbox_to_int_tuple(frame_bbox),
            "pre_heatmap": current_heatmap,
            "pre_entropy": float(compute_normalized_heatmap_entropy(current_heatmap)),
            "localization_strategy": "iterative_local",
            "fusion_mode": "none_last_local_only",
        }
        stop_ratio = component_short_ratio(stage)
        stage["adaptive_stop_component_short_ratio"] = float(stop_ratio)
        if isinstance(selected_component, dict) and adaptive_stop_hotspot_ratio is not None:
            if stop_ratio >= float(adaptive_stop_hotspot_ratio):
                stage["adaptive_stop_triggered"] = True
                stage["adaptive_stop_stopped_before_local_heatmap"] = True
                stage["stop_reason"] = "hotspot_fills_candidate"
                stage["final_bbox_semantics"] = selection.get(
                    "bbox_semantics", "current_centered_square_crop"
                )
                stages.append(stage)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

        if is_terminal_stage:
            stage["terminal_crop"] = True
            stage["local_heatmap_skipped"] = "no_later_stage_uses_it"
            stage["stop_reason"] = "min_crop_side"
            stage["final_bbox_semantics"] = selection.get(
                "bbox_semantics", "current_centered_square_crop"
            )
            stages.append(stage)
            break

        crop = image.crop(crop_bbox)
        with torch.no_grad():
            local_heatmap = adapter.heatmap_fn(
                crop,
                attention_prompt,
                adapter.general_prompt,
                adapter.model,
                adapter.processor,
            )
        local_heatmap = sanitize_heatmap(local_heatmap)
        stage.update(
            {
                "local_heatmap": local_heatmap,
                "local_heatmap_shape": [int(v) for v in local_heatmap.shape],
                "local_entropy": float(compute_normalized_heatmap_entropy(local_heatmap)),
            }
        )
        stages.append(stage)
        frame_image = crop
        frame_bbox = crop_bbox
        current_heatmap = local_heatmap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return stages


def run_localization_chain(*, strategy: str, **kwargs: Any) -> list[dict[str, Any]]:
    stage_candidate_scorer = kwargs.pop("stage_candidate_scorer", None)
    local_heatmap_fn = kwargs.pop("local_heatmap_fn", None)
    if strategy == "phf":
        return run_top1_hotspot_chain(
            stage_candidate_scorer=stage_candidate_scorer,
            local_heatmap_fn=local_heatmap_fn,
            **kwargs,
        )
    if stage_candidate_scorer is not None or local_heatmap_fn is not None:
        raise ValueError(
            "custom stage selectors and local heatmap functions are only supported by phf"
        )
    if strategy == "iterative_local":
        return run_iterative_local_chain(**kwargs)
    raise ValueError(
        f"unsupported localization strategy: {strategy!r}; "
        f"expected one of {LOCALIZATION_STRATEGIES}"
    )
