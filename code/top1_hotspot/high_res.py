from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PIL import Image

from .geometry import resize_heatmap, sanitize_heatmap


def split_image_tiles(
    image: Image.Image,
    *,
    tile_size: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_width, image_height = image.size
    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size!r}")

    grid_cols = int(np.ceil(float(image_width) / float(tile_size)))
    grid_rows = int(np.ceil(float(image_height) / float(tile_size)))
    grid_cols = max(1, grid_cols)
    grid_rows = max(1, grid_rows)
    step_w = int(np.ceil(float(image_width) / float(grid_cols)))
    step_h = int(np.ceil(float(image_height) / float(grid_rows)))

    tiles: list[dict[str, Any]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = min(image_width, col * step_w)
            y1 = min(image_height, row * step_h)
            x2 = min(image_width, (col + 1) * step_w)
            y2 = min(image_height, (row + 1) * step_h)
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = (int(x1), int(y1), int(x2), int(y2))
            tiles.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "bbox": bbox,
                    "image": image.crop(bbox),
                }
            )

    info = {
        "tile_size": int(tile_size),
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "tile_count": int(len(tiles)),
        "step_size": [int(step_w), int(step_h)],
        "image_size": [int(image_width), int(image_height)],
    }
    return tiles, info


def tile_heatmap_mosaic(
    map_func: Callable[[Image.Image, str, str, Any, Any], Any],
    image: Image.Image,
    prompt: str,
    general_prompt: str,
    model: Any,
    processor: Any,
    *,
    tile_size: int = 1024,
) -> tuple[np.ndarray, dict[str, Any]]:
    tiles, info = split_image_tiles(image, tile_size=tile_size)
    if not tiles:
        raise RuntimeError("tile_heatmap_mosaic produced no image tiles")

    heatmaps_by_row: list[list[np.ndarray]] = [
        [] for _ in range(int(info["grid_rows"]))
    ]
    tile_records: list[dict[str, Any]] = []
    scale_y_values: list[float] = []
    scale_x_values: list[float] = []
    for tile in tiles:
        heatmap = sanitize_heatmap(
            map_func(tile["image"], prompt, general_prompt, model, processor)
        )
        heatmaps_by_row[int(tile["row"])].append(heatmap)
        x1, y1, x2, y2 = [int(v) for v in tile["bbox"]]
        scale_y_values.append(float(heatmap.shape[0]) / float(max(1, y2 - y1)))
        scale_x_values.append(float(heatmap.shape[1]) / float(max(1, x2 - x1)))
        tile_records.append(
            {
                "row": int(tile["row"]),
                "col": int(tile["col"]),
                "bbox": [x1, y1, x2, y2],
                "tile_size": [int(x2 - x1), int(y2 - y1)],
                "local_heatmap_shape": [int(v) for v in heatmap.shape],
                "local_heatmap_sum": float(heatmap.sum()),
            }
        )

    scale_y = float(np.median(scale_y_values))
    scale_x = float(np.median(scale_x_values))
    image_width, image_height = image.size
    mosaic_height = max(1, int(round(float(image_height) * scale_y)))
    mosaic_width = max(1, int(round(float(image_width) * scale_x)))
    mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)

    tile_index = 0
    for tile in tiles:
        heatmap = heatmaps_by_row[int(tile["row"])][int(tile["col"])]
        x1, y1, x2, y2 = [int(v) for v in tile["bbox"]]
        hx1 = int(round(float(x1) / float(image_width) * float(mosaic_width)))
        hy1 = int(round(float(y1) / float(image_height) * float(mosaic_height)))
        hx2 = int(round(float(x2) / float(image_width) * float(mosaic_width)))
        hy2 = int(round(float(y2) / float(image_height) * float(mosaic_height)))
        hx1 = max(0, min(mosaic_width - 1, hx1))
        hy1 = max(0, min(mosaic_height - 1, hy1))
        hx2 = max(hx1 + 1, min(mosaic_width, hx2))
        hy2 = max(hy1 + 1, min(mosaic_height, hy2))
        resized = resize_heatmap(heatmap, (hy2 - hy1, hx2 - hx1))
        mosaic[hy1:hy2, hx1:hx2] = resized
        tile_records[tile_index]["mosaic_window"] = [int(hx1), int(hy1), int(hx2), int(hy2)]
        tile_index += 1

    mosaic = mosaic.astype(np.float32)
    metadata = {
        **info,
        "mode": "tile1024",
        "normalization": "resize_to_image_proportional_mosaic",
        "triggered": True,
        "mosaic_heatmap_shape": [int(v) for v in mosaic.shape],
        "mosaic_scale": [float(scale_x), float(scale_y)],
        "tile_boxes": tile_records,
    }
    return sanitize_heatmap(mosaic), metadata
