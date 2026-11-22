from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import (
    bbox_to_int_tuple,
    center_square_bbox,
    heatmap_window_to_image_bbox,
    max_heat_square_bbox,
    sanitize_heatmap,
)


def topmass_mask(heatmap: Any, mass_fraction: float) -> dict[str, Any]:
    if not 0.0 < float(mass_fraction) <= 1.0:
        raise ValueError(f"topmass must be in (0, 1], got {mass_fraction!r}")

    array = sanitize_heatmap(heatmap)
    positive = array > 0
    total_mass = float(array[positive].sum())
    if total_mass <= 0.0:
        return {
            "mask": np.zeros_like(array, dtype=bool),
            "threshold": 0.0,
            "total_mass": 0.0,
            "covered_mass": 0.0,
            "covered_fraction": 0.0,
            "mask_pixels": 0,
        }

    flat = array.reshape(-1)
    order = np.argsort(flat)[::-1]
    ordered_values = flat[order]
    positive_ordered = ordered_values > 0
    order = order[positive_ordered]
    ordered_values = ordered_values[positive_ordered]
    cumulative = np.cumsum(ordered_values, dtype=np.float64)
    target_mass = float(total_mass * float(mass_fraction))
    cutoff = int(np.searchsorted(cumulative, target_mass, side="left"))
    cutoff = max(0, min(cutoff, len(ordered_values) - 1))
    threshold = float(ordered_values[cutoff])
    mask = (array >= threshold) & positive
    covered_mass = float(array[mask].sum())
    return {
        "mask": mask,
        "threshold": threshold,
        "total_mass": total_mass,
        "covered_mass": covered_mass,
        "covered_fraction": covered_mass / max(total_mass, 1e-12),
        "mask_pixels": int(mask.sum()),
    }


def connected_components(
    mask: np.ndarray,
    heatmap: Any,
) -> list[dict[str, Any]]:
    array = sanitize_heatmap(heatmap)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    neighbors = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask[start_y, start_x]:
                continue

            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                for dx, dy in neighbors:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))

            area = len(xs)
            xs_array = np.asarray(xs, dtype=np.int64)
            ys_array = np.asarray(ys, dtype=np.int64)
            values = array[ys_array, xs_array].astype(np.float64)
            mass = float(values.sum())
            if mass <= 0.0:
                continue
            mean_value = float(values.mean())
            peak_value = float(values.max())
            cx = float(((xs_array.astype(np.float64) + 0.5) * values).sum() / mass)
            cy = float(((ys_array.astype(np.float64) + 0.5) * values).sum() / mass)
            bbox = (
                int(xs_array.min()),
                int(ys_array.min()),
                int(xs_array.max() + 1),
                int(ys_array.max() + 1),
            )
            score = float(peak_value * mean_value * math.log1p(area))
            components.append(
                {
                    "bbox": bbox,
                    "centroid_heatmap_xy": [cx, cy],
                    "area": int(area),
                    "mass": mass,
                    "mean": mean_value,
                    "peak": peak_value,
                    "score": score,
                }
            )

    components.sort(key=lambda item: float(item["score"]), reverse=True)
    return components


def hotspot_component_square_bbox(
    heatmap: Any,
    image_size: tuple[int, int],
    side: int,
    *,
    topmass: float,
    component_score: str = "default",
) -> dict[str, Any]:
    array = sanitize_heatmap(heatmap)
    heatmap_height, heatmap_width = array.shape
    mask_info = topmass_mask(array, topmass)
    components = connected_components(
        mask_info["mask"],
        array,
    )
    if component_score != "default":
        scores = {
            "mass": lambda component: component["mass"],
            "peak_mean": lambda component: component["peak"] * component["mean"],
        }
        score = scores[component_score]
        for component in components:
            component["score"] = score(component)
        components.sort(key=lambda component: component["score"], reverse=True)
    if not components:
        fallback = max_heat_square_bbox(array, image_size, side)
        return {
            "bbox": bbox_to_int_tuple(fallback["bbox"]),
            "side": int(side),
            "mode": "fallback_max_sum",
            "fallback_reason": "no_component",
            "mask": {key: value for key, value in mask_info.items() if key != "mask"},
            "component_count": 0,
            "selected_component": None,
            "runner_up_component": None,
            "score": float(fallback["score"]),
        }

    selected = components[0]
    cx_heatmap, cy_heatmap = selected["centroid_heatmap_xy"]
    cx_image = float(cx_heatmap) / float(heatmap_width) * float(image_size[0])
    cy_image = float(cy_heatmap) / float(heatmap_height) * float(image_size[1])
    bbox = center_square_bbox((cx_image, cy_image), int(side), image_size)

    serial_components = []
    for component in components[:2]:
        serial = dict(component)
        serial["bbox"] = [int(v) for v in component["bbox"]]
        serial["image_bbox"] = heatmap_window_to_image_bbox(
            component["bbox"],
            image_size=image_size,
            heatmap_shape=array.shape,
        )
        serial_components.append(serial)

    selected_serial = serial_components[0]
    return {
        "bbox": bbox_to_int_tuple(bbox),
        "side": int(side),
        "mode": "hotspot_component_center",
        "fallback_reason": None,
        "mask": {key: value for key, value in mask_info.items() if key != "mask"},
        "component_count": len(components),
        "selected_component": selected_serial,
        "runner_up_component": serial_components[1] if len(serial_components) > 1 else None,
        "score": float(selected["score"]),
    }


def ranked_hotspot_square_bboxes(
    heatmap: Any,
    image_size: tuple[int, int],
    side: int,
    *,
    topmass: float,
    top_k: int,
) -> dict[str, Any]:
    """Return distinct square crops centered on the top-ranked hotspot components."""
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k!r}")
    array = sanitize_heatmap(heatmap)
    mask_info = topmass_mask(array, topmass)
    components = connected_components(mask_info["mask"], array)
    heatmap_height, heatmap_width = array.shape
    selected: list[dict[str, Any]] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()

    def overlap_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        return float(intersection) / float(max(1, first_area + second_area - intersection))

    def accept_bbox(bbox: tuple[int, int, int, int]) -> bool:
        if bbox in seen_bboxes:
            return False
        return all(overlap_iou(bbox, tuple(item["crop_bbox"])) < 0.5 for item in selected)

    for component in components:
        cx_heatmap, cy_heatmap = component["centroid_heatmap_xy"]
        center = (
            float(cx_heatmap) / float(heatmap_width) * float(image_size[0]),
            float(cy_heatmap) / float(heatmap_height) * float(image_size[1]),
        )
        bbox = bbox_to_int_tuple(center_square_bbox(center, int(side), image_size))
        if not accept_bbox(bbox):
            continue
        seen_bboxes.add(bbox)
        serial = dict(component)
        serial["bbox"] = [int(value) for value in component["bbox"]]
        serial["image_bbox"] = heatmap_window_to_image_bbox(
            component["bbox"], image_size=image_size, heatmap_shape=array.shape
        )
        serial["crop_bbox"] = bbox
        selected.append(serial)
        if len(selected) >= int(top_k):
            break
    if len(selected) < int(top_k):
        flat_order = np.argsort(array.reshape(-1))[::-1]
        for flat_index in flat_order:
            y, x = np.unravel_index(int(flat_index), array.shape)
            if float(array[y, x]) <= 0.0:
                break
            center = (
                (float(x) + 0.5) / float(heatmap_width) * float(image_size[0]),
                (float(y) + 0.5) / float(heatmap_height) * float(image_size[1]),
            )
            bbox = bbox_to_int_tuple(center_square_bbox(center, int(side), image_size))
            if not accept_bbox(bbox):
                continue
            seen_bboxes.add(bbox)
            selected.append(
                {
                    "crop_bbox": bbox,
                    "score": float(array[y, x]),
                    "fallback_reason": "fill_from_ranked_heatmap_peaks",
                    "peak_heatmap_xy": [int(x), int(y)],
                }
            )
            if len(selected) >= int(top_k):
                break
    if not selected:
        fallback = max_heat_square_bbox(array, image_size, side)
        selected.append(
            {
                "crop_bbox": bbox_to_int_tuple(fallback["bbox"]),
                "score": float(fallback["score"]),
                "fallback_reason": "no_component",
            }
        )
    return {
        "mode": "ranked_hotspot_components",
        "requested_top_k": int(top_k),
        "selected_count": len(selected),
        "side": int(side),
        "mask": {key: value for key, value in mask_info.items() if key != "mask"},
        "selected": selected,
    }
