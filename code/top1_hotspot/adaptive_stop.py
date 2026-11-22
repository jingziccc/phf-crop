from __future__ import annotations

from typing import Any

from .geometry import bbox_to_int_tuple


def selected_component_bbox(stage: dict[str, Any]) -> tuple[int, int, int, int] | None:
    selected = stage.get("selection", {}).get("selected_component")
    if not isinstance(selected, dict):
        return None
    bbox = selected.get("image_bbox")
    if bbox is None:
        return None
    return bbox_to_int_tuple(bbox)


def component_short_ratio(stage: dict[str, Any]) -> float:
    bbox = selected_component_bbox(stage)
    candidate_bbox = stage.get("crop_bbox")
    if candidate_bbox is not None:
        candidate_x1, candidate_y1, candidate_x2, candidate_y2 = bbox_to_int_tuple(
            candidate_bbox
        )
        candidate_short_side = min(
            max(0, candidate_x2 - candidate_x1),
            max(0, candidate_y2 - candidate_y1),
        )
    else:
        candidate_short_side = int(stage.get("side") or 0)
    if bbox is None or candidate_short_side <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return float(min(width, height)) / float(candidate_short_side)


def choose_adaptive_stop_stage(
    stages: list[dict[str, Any]],
    hotspot_ratio: float,
) -> tuple[dict[str, Any], float, bool]:
    if not stages:
        raise RuntimeError("adaptive stop requires at least one PHF-Crop stage")
    for stage in stages:
        ratio = component_short_ratio(stage)
        if ratio >= float(hotspot_ratio):
            return stage, ratio, True
    final = stages[-1]
    return final, component_short_ratio(final), False
