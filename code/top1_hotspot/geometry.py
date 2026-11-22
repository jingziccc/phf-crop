from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from PIL import Image


HALVING_INITIAL_AREA_RATIO = 0.5
HALVING_RATIO_DECAY = 0.5
MAX_HALVING_STAGES = 64


def sanitize_heatmap(heatmap: Any) -> np.ndarray:
    array = np.asarray(heatmap, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(array, 0.0)


def normalize_heatmap_sum(heatmap: Any) -> np.ndarray:
    array = sanitize_heatmap(heatmap)
    total = float(array.sum())
    if total > 0.0:
        return array / total
    return array


def bbox_to_int_tuple(bbox: Any) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def center_square_bbox(
    center: tuple[float, float],
    side: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    side = int(max(1, min(side, image_width, image_height)))
    cx, cy = center
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x1 = max(0, min(image_width - side, x1))
    y1 = max(0, min(image_height - side, y1))
    return (int(x1), int(y1), int(x1 + side), int(y1 + side))


def heatmap_window_to_image_bbox(
    window: tuple[int, int, int, int] | list[int],
    image_size: tuple[int, int],
    heatmap_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    hx1, hy1, hx2, hy2 = [int(v) for v in window]
    image_width, image_height = image_size
    heatmap_height, heatmap_width = heatmap_shape
    x1 = int(round(float(hx1) / float(heatmap_width) * float(image_width)))
    y1 = int(round(float(hy1) / float(heatmap_height) * float(image_height)))
    x2 = int(round(float(hx2) / float(heatmap_width) * float(image_width)))
    y2 = int(round(float(hy2) / float(heatmap_height) * float(image_height)))
    return bbox_to_int_tuple(
        (
            max(0, min(image_width, x1)),
            max(0, min(image_height, y1)),
            max(0, min(image_width, x2)),
            max(0, min(image_height, y2)),
        )
    )


def image_bbox_to_heatmap_window(
    bbox: Any,
    image_size: tuple[int, int],
    heatmap_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    image_width, image_height = image_size
    heatmap_height, heatmap_width = heatmap_shape
    hx1 = int(np.floor(float(x1) / float(image_width) * float(heatmap_width)))
    hy1 = int(np.floor(float(y1) / float(image_height) * float(heatmap_height)))
    hx2 = int(np.ceil(float(x2) / float(image_width) * float(heatmap_width)))
    hy2 = int(np.ceil(float(y2) / float(image_height) * float(heatmap_height)))
    hx1 = max(0, min(heatmap_width, hx1))
    hy1 = max(0, min(heatmap_height, hy1))
    hx2 = max(hx1 + 1, min(heatmap_width, hx2))
    hy2 = max(hy1 + 1, min(heatmap_height, hy2))
    return (int(hx1), int(hy1), int(hx2), int(hy2))


def resize_heatmap(heatmap: Any, output_shape: tuple[int, int]) -> np.ndarray:
    output_height, output_width = [int(v) for v in output_shape]
    if output_height <= 0 or output_width <= 0:
        raise ValueError(f"invalid output heatmap shape: {output_shape}")
    array = sanitize_heatmap(heatmap)
    heatmap_image = Image.fromarray(array, mode="F")
    resized = heatmap_image.resize((output_width, output_height), Image.Resampling.BICUBIC)
    return sanitize_heatmap(resized)


def _integral_image(heatmap: np.ndarray) -> np.ndarray:
    integral = np.zeros((heatmap.shape[0] + 1, heatmap.shape[1] + 1), dtype=np.float32)
    integral[1:, 1:] = heatmap.cumsum(axis=0).cumsum(axis=1)
    return integral


def _integral_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _heatmap_window_to_centered_square_bbox(
    window: tuple[int, int, int, int],
    side: int,
    image_size: tuple[int, int],
    heatmap_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    hx1, hy1, hx2, hy2 = window
    image_width, image_height = image_size
    heatmap_height, heatmap_width = heatmap_shape
    cx = ((float(hx1) + float(hx2)) / 2.0) / float(heatmap_width) * float(image_width)
    cy = ((float(hy1) + float(hy2)) / 2.0) / float(heatmap_height) * float(image_height)
    return center_square_bbox((cx, cy), side, image_size)


def max_heat_square_bbox(
    heatmap: Any,
    image_size: tuple[int, int],
    side: int,
) -> dict[str, Any]:
    array = sanitize_heatmap(heatmap)
    heatmap_height, heatmap_width = array.shape
    image_width, image_height = image_size
    side = int(max(1, min(side, image_width, image_height)))

    window_width = int(math.ceil(float(side) / float(image_width) * float(heatmap_width)))
    window_height = int(math.ceil(float(side) / float(image_height) * float(heatmap_height)))
    window_width = max(1, min(heatmap_width, window_width))
    window_height = max(1, min(heatmap_height, window_height))

    integral = _integral_image(array)
    best_score = -float("inf")
    best_window = (0, 0, window_width, window_height)
    candidate_count = 0
    for y in range(0, heatmap_height - window_height + 1):
        for x in range(0, heatmap_width - window_width + 1):
            candidate_count += 1
            score = _integral_sum(integral, x, y, x + window_width, y + window_height)
            if score > best_score:
                best_score = score
                best_window = (x, y, x + window_width, y + window_height)

    bbox = _heatmap_window_to_centered_square_bbox(
        best_window,
        side,
        image_size=image_size,
        heatmap_shape=array.shape,
    )
    return {
        "bbox": bbox,
        "side": int(side),
        "score": float(best_score),
        "heatmap_window": tuple(int(v) for v in best_window),
        "heatmap_window_size": [int(window_width), int(window_height)],
        "candidate_count": int(candidate_count),
    }


def build_stage_targets(
    image_size: tuple[int, int],
    min_crop_side: int,
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    image_area = float(image_width * image_height)
    short_side = min(image_width, image_height)
    final_side = int(max(1, min(min_crop_side, short_side)))
    targets: list[dict[str, Any]] = []

    ratio = HALVING_INITIAL_AREA_RATIO
    for _ in range(MAX_HALVING_STAGES):
        raw_side = math.sqrt(image_area * float(ratio))
        if raw_side < float(min_crop_side):
            break
        targets.append(
            {
                "kind": "area_ratio",
                "ratio": float(ratio),
                "raw_side": float(raw_side),
                "side": int(round(min(raw_side, short_side))),
            }
        )
        ratio *= HALVING_RATIO_DECAY

    if not targets or int(targets[-1]["side"]) != final_side:
        targets.append(
            {
                "kind": "min_crop_side",
                "ratio": None,
                "raw_side": float(min_crop_side),
                "side": int(final_side),
            }
        )
    return targets


def target_label(target: dict[str, Any], min_crop_side: int) -> str:
    if target["ratio"] is None:
        return f"{int(min_crop_side)}px"
    percent = float(target["ratio"]) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}%"
    return f"{percent:.3f}".rstrip("0").rstrip(".") + "%"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def dumps_json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, indent=2)
