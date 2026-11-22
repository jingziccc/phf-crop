from __future__ import annotations

import base64
import html
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib import patches
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]


def bootstrap_repo(repo: Path | str = REPO_ROOT) -> Path:
    repo = Path(repo).resolve()
    os.chdir(repo)
    for path in (
        repo,
        repo / "visualization",
        repo / "code",
        repo / "code" / "runtime",
        repo / "code" / "scripts",
    ):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return repo


bootstrap_repo(REPO_ROOT)

from top1_hotspot.components import connected_components, topmass_mask  # noqa: E402
from top1_hotspot.config import MODEL_DEFAULTS  # noqa: E402
from top1_hotspot.geometry import (  # noqa: E402
    bbox_to_int_tuple,
    image_bbox_to_heatmap_window,
    sanitize_heatmap,
)
from phf_crop_visualization_utils import (  # noqa: E402
    VizConfig,
    build_context,
    choose_sample,
    display_answer,
    heatmap_for_display,
    load_adapter_for_config,
    run_visual_trace,
)


BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
RED = "#FF0000"
RED_DASH = (0, (2.0, 1.2))
WHITE = "#FFFFFF"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
DARK = "#111827"
SELECTED_BOX_COLOR = RED
STAGE_BOX_COLORS = (
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#6B7280",
)
COMPONENT_COLORS = (
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#6B7280",
    "#8B5CF6",
    "#64748B",
    "#0F766E",
)


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    color: str,
    width: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1, x2, y2 = xy

    def draw_dashed_line(start: tuple[int, int], end: tuple[int, int]) -> None:
        sx, sy = start
        ex, ey = end
        length = max(abs(ex - sx), abs(ey - sy))
        if length <= 0:
            return
        step = max(1, dash + gap)
        for offset in range(0, length + 1, step):
            segment_end = min(length, offset + dash)
            t1 = offset / length
            t2 = segment_end / length
            draw.line(
                (
                    int(round(sx + (ex - sx) * t1)),
                    int(round(sy + (ey - sy) * t1)),
                    int(round(sx + (ex - sx) * t2)),
                    int(round(sy + (ey - sy) * t2)),
                ),
                fill=color,
                width=width,
            )

    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))


def times_new_roman_font(size: int) -> ImageFont.ImageFont:
    for font in (
        "Times New Roman.ttf",
        "Times New Roman",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/croscore/Tinos-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(font, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class LoopPanelConfig:
    repo: Path = REPO_ROOT
    model: str = "qwen2_5"
    task: str = "hrbench_8k"
    gpu: str = "2"
    sample_index: int = 8
    sample_id: str | None = None
    topmass: float | None = None
    initial_heatmap_bootstrap: str = "tile1024"
    adaptive_stop_hotspot_ratio: float | None = 1.0
    localization_strategy: str = "phf"
    max_loops: int | None = None
    use_prediction_cache: bool = False
    generate_answers: bool = True
    save: bool = False
    save_dir: Path = REPO_ROOT / "outputs" / "visualization" / "loop_panels"
    dpi: int = 220
    display_dpi: int = 160
    fusion_overview_resolution: int = 1024
    fusion_surface_resolution: int = 256
    web_texture_resolution: int = 2048
    web_scene_width: int = 1800
    web_scene_height: int = 980
    figsize_3: tuple[float, float] = (12.0, 4.0)
    figsize_4: tuple[float, float] = (15.5, 4.0)
    max_components: int = 3
    show_text: bool = False
    fusion_overview_figsize: tuple[float, float] = (12.0, 6.4)
    fusion_overview_elev: float = 12.0
    fusion_overview_azim: float = -66.0


def build_realtime_trace(config: LoopPanelConfig) -> tuple[dict[str, Any], Any, Any]:
    viz_config = VizConfig(
        repo=Path(config.repo),
        model=config.model,
        task=config.task,
        gpu=str(config.gpu),
        sample_index=int(config.sample_index),
        sample_id=config.sample_id,
        topmass=config.topmass,
        initial_heatmap_bootstrap=config.initial_heatmap_bootstrap,
        adaptive_stop_hotspot_ratio=config.adaptive_stop_hotspot_ratio,
        localization_strategy=config.localization_strategy,
        answer_mode="realtime" if config.generate_answers else "prediction_then_realtime",
        use_prediction_cache=bool(config.use_prediction_cache),
        save_dir=Path(config.save_dir),
    )
    context = build_context(viz_config)
    adapter, _device = load_adapter_for_config(viz_config)
    row = choose_sample(
        context.rows,
        sample_id=viz_config.sample_id,
        sample_index=viz_config.sample_index,
    )
    trace = run_visual_trace(
        row,
        context=context,
        adapter=adapter,
        generate_answers=bool(config.generate_answers),
    )
    return trace, context, adapter


def selected_component(stage: dict[str, Any]) -> dict[str, Any] | None:
    selected = stage.get("selection", {}).get("selected_component")
    return selected if isinstance(selected, dict) else None


def selection_components(stage: dict[str, Any]) -> list[dict[str, Any]]:
    components = stage.get("selection", {}).get("components")
    if not isinstance(components, list):
        return []
    return [item for item in components if isinstance(item, dict) and item.get("bbox")]


def adaptive_stop_stage_index(trace: dict[str, Any]) -> int:
    stages = trace.get("stages") or []
    stop = trace.get("adaptive_stop") or {}
    selected_stage = stop.get("selected_stage")
    if selected_stage is None:
        return len(stages) - 1
    for idx, stage in enumerate(stages):
        if int(stage.get("stage") or idx + 1) == int(selected_stage):
            return idx
    return min(len(stages) - 1, max(0, int(selected_stage) - 1))


def loop_count(trace: dict[str, Any], config: LoopPanelConfig) -> int:
    stages = trace.get("stages") or []
    if not stages:
        return 0
    # Loop 0 selects stage 1 from the initial heatmap. Loop k>0 shows stage k
    # local heatmap/fusion, then selects stage k+1.
    count = adaptive_stop_stage_index(trace) + 1
    if config.max_loops is not None:
        count = min(count, int(config.max_loops))
    return max(1, min(count, len(stages)))


def stage_label(stage: dict[str, Any]) -> str:
    target = str(stage.get("target_label") or "")
    return f"S{stage.get('stage', '?')} {target}".strip()


def crop_image_for_stage(image: Image.Image, stage: dict[str, Any]) -> Image.Image:
    return image.crop(bbox_to_int_tuple(stage["crop_bbox"]))


def current_image_and_heatmap(
    trace: dict[str, Any],
    loop_index: int,
) -> tuple[Image.Image, np.ndarray, str]:
    image = trace["image"]
    if loop_index == 0:
        return image, sanitize_heatmap(trace["initial_heatmap"]), "original image / initial heatmap"
    previous_stage = trace["stages"][loop_index - 1]
    return (
        crop_image_for_stage(image, previous_stage),
        sanitize_heatmap(previous_stage.get("local_heatmap", trace["initial_heatmap"])),
        f"{stage_label(previous_stage)} crop / local heatmap",
    )


def fused_heatmap_for_loop(trace: dict[str, Any], loop_index: int) -> np.ndarray | None:
    if loop_index == 0:
        return None
    previous_stage = trace["stages"][loop_index - 1]
    fused_heatmap = previous_stage.get("fused_heatmap")
    return sanitize_heatmap(fused_heatmap) if fused_heatmap is not None else None


def final_heatmap_for_trace(trace: dict[str, Any], stages: list[dict[str, Any]]) -> np.ndarray:
    if stages:
        for key in ("fused_heatmap", "pre_heatmap"):
            heatmap = stages[-1].get(key)
            if heatmap is not None:
                return sanitize_heatmap(heatmap)
    return sanitize_heatmap(trace["initial_heatmap"])


def selection_stage_for_loop(trace: dict[str, Any], loop_index: int) -> dict[str, Any]:
    stages = trace["stages"]
    return stages[min(loop_index, len(stages) - 1)]


def heatmap_used_for_selection(trace: dict[str, Any], loop_index: int) -> np.ndarray:
    stage = selection_stage_for_loop(trace, loop_index)
    return sanitize_heatmap(stage["pre_heatmap"])


def hide_axis(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_image(ax: Any, image: Image.Image, title: str | None = None) -> None:
    width, height = image.size
    ax.imshow(image, extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    hide_axis(ax)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def show_image_heatmap_overlay(
    ax: Any,
    image: Image.Image,
    heatmap: Any,
    *,
    alpha: float = 0.45,
    title: str | None = None,
) -> None:
    width, height = image.size
    ax.imshow(image, extent=(0, width, height, 0))
    ax.imshow(
        heatmap_for_display(heatmap),
        cmap="inferno",
        alpha=float(alpha),
        extent=(0, width, height, 0),
        interpolation="bicubic",
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    hide_axis(ax)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def image_size_text(image: Image.Image) -> str:
    return f"{int(image.size[0])} x {int(image.size[1])} px"


def heatmap_size_text(heatmap: Any) -> str:
    array = sanitize_heatmap(heatmap)
    return f"{int(array.shape[1])} x {int(array.shape[0])} values"


def show_heatmap_values(
    ax: Any,
    heatmap: Any,
    title: str | None = None,
) -> None:
    array = heatmap_for_display(heatmap)
    height, width = array.shape
    ax.imshow(
        array,
        cmap="inferno",
        interpolation="nearest",
        extent=(0, width, height, 0),
    )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    hide_axis(ax)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def draw_heatmap_rect(
    ax: Any,
    bbox: Any,
    color: str,
    *,
    label: str | None = None,
    linewidth: float = 1.8,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    rect = patches.Rectangle(
        (x1, y1),
        max(1, x2 - x1),
        max(1, y2 - y1),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            max(0, y1 - 0.8),
            label,
            fontsize=7,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.9, "pad": 1.4},
        )


def draw_image_rect(
    ax: Any,
    bbox: Any,
    color: str,
    *,
    label: str | None = None,
    linewidth: float = 2.2,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    x1, y1, x2, y2 = bbox_to_int_tuple(bbox)
    rect = patches.Rectangle(
        (x1, y1),
        max(1, x2 - x1),
        max(1, y2 - y1),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            max(0, y1 - 3),
            label,
            fontsize=8,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.9, "pad": 1.4},
        )


def component_color(index: int, *, is_top1: bool) -> str:
    if is_top1:
        return BLUE
    return COMPONENT_COLORS[int(index) % len(COMPONENT_COLORS)]


def draw_topmass_heatmap(
    ax: Any,
    trace: dict[str, Any],
    context: Any,
    loop_index: int,
    *,
    max_components: int = 12,
    show_text: bool = False,
) -> None:
    image = trace["image"]
    heatmap = heatmap_used_for_selection(trace, loop_index)
    stage = selection_stage_for_loop(trace, loop_index)
    selected = selected_component(stage)
    selected_bbox = tuple(selected.get("bbox")) if selected and selected.get("bbox") else None

    # Show the original heatmap values, then draw the topmass connected
    # top-1 component selected from that heatmap.
    heatmap_height, heatmap_width = heatmap.shape
    ax.imshow(
        heatmap_for_display(heatmap),
        cmap="inferno",
        interpolation="nearest",
        extent=(0, heatmap_width, heatmap_height, 0),
    )

    if loop_index == 0:
        for component in selection_components(stage):
            component_bbox = tuple(component.get("bbox"))
            if selected_bbox is not None and bbox_to_int_tuple(component_bbox) == bbox_to_int_tuple(selected_bbox):
                continue
            draw_heatmap_rect(
                ax,
                component_bbox,
                WHITE,
                linewidth=1.1,
                linestyle=RED_DASH,
                alpha=0.95,
            )

    if selected and selected_bbox is not None:
        score = float(selected.get("score", 0.0))
        draw_heatmap_rect(
            ax,
            selected_bbox,
            RED,
            label=f"top1 {score:.2g}" if show_text else None,
            linewidth=1.5,
            linestyle=RED_DASH,
            alpha=1.0,
        )

    crop_window = image_bbox_to_heatmap_window(
        stage["crop_bbox"],
        image_size=image.size,
        heatmap_shape=heatmap.shape,
    )
    draw_heatmap_rect(
        ax,
        crop_window,
        RED,
        label=f"expanded crop {stage.get('target_label', '')}" if show_text else None,
        linewidth=1.6,
        linestyle="-",
    )

    ax.set_xlim(0, heatmap_width)
    ax.set_ylim(heatmap_height, 0)
    hide_axis(ax)
    if show_text:
        title = f"heatmap + top1 component ({heatmap_width} x {heatmap_height}) -> {stage_label(stage)}"
        ax.set_title(title, fontsize=10, loc="left")


def figure_name(trace: dict[str, Any], loop_index: int, panel_name: str = "panel") -> str:
    sample_id = str(trace["row"].get("id", "sample"))
    return f"{sample_id}_loop_{loop_index:02d}_{panel_name}.png"


def save_or_display(
    fig: plt.Figure,
    trace: dict[str, Any],
    config: LoopPanelConfig,
    loop_index: int,
    panel_name: str = "panel",
) -> None:
    fig.set_dpi(int(config.display_dpi))
    fig.tight_layout()
    if config.save:
        save_dir = Path(config.save_dir)
        if not save_dir.is_absolute():
            save_dir = REPO_ROOT / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / figure_name(trace, loop_index, panel_name)
        fig.savefig(out_path, dpi=int(config.dpi), bbox_inches="tight", pad_inches=0)
        print(f"saved: {out_path}")
    display(fig)
    plt.close(fig)


def plot_single_image_panel(image: Image.Image, title: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    show_image(ax, image, title)
    return fig


def plot_single_original_next_crop_panel(
    trace: dict[str, Any],
    loop_index: int,
    title: str | None = None,
) -> plt.Figure:
    image = trace["image"]
    stage = selection_stage_for_loop(trace, loop_index)
    heatmap = heatmap_used_for_selection(trace, loop_index)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    show_image_heatmap_overlay(ax, image, heatmap, title=title)
    draw_image_rect(
        ax,
        stage["crop_bbox"],
        RED,
        label=f"next crop {stage.get('target_label', '')}" if title else None,
        linewidth=1.7,
        linestyle="-",
    )
    return fig


def plot_single_heatmap_panel(heatmap: Any, title: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    show_heatmap_values(ax, heatmap, title)
    return fig


def plot_single_top1_panel(
    trace: dict[str, Any],
    context: Any,
    config: LoopPanelConfig,
    loop_index: int,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    draw_topmass_heatmap(
        ax,
        trace,
        context,
        loop_index,
        max_components=config.max_components,
        show_text=config.show_text,
    )
    return fig


def _resample_heatmap_for_surface(heatmap: Any, texture_side: int) -> np.ndarray:
    array = heatmap_for_display(heatmap)
    height, width = array.shape
    texture_side = max(2, int(texture_side))
    scale = float(texture_side) / max(height, width)
    if abs(scale - 1.0) < 1e-6:
        return array
    out_width = max(2, int(round(width * scale)))
    out_height = max(2, int(round(height * scale)))
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(image.resize((out_width, out_height), Image.Resampling.BICUBIC))


def _draw_global_coordinate_frame(ax: Any, depth: float, image_size: tuple[int, int]) -> None:
    width, height = image_size
    ax.plot(
        [depth] * 5,
        [0, width, width, 0, 0],
        [0, 0, height, height, 0],
        color=(0.30, 0.33, 0.37, 0.34),
        linewidth=0.65,
    )


def _draw_heatmap_surface(
    ax: Any,
    heatmap: Any,
    *,
    depth: float,
    image_bbox: Any,
    image_height: int,
    texture_side: int,
    edgecolor: str,
    linewidth: float,
) -> None:
    array = _resample_heatmap_for_surface(heatmap, texture_side=texture_side)
    height, width = array.shape
    x1, y1, x2, y2 = bbox_to_int_tuple(image_bbox)
    global_x = np.linspace(x1, x2, width)
    # Image bboxes use a top-left origin; the upright 3D plane uses bottom-left.
    global_y = np.linspace(image_height - y1, image_height - y2, height)
    grid_x, grid_y = np.meshgrid(global_x, global_y)
    grid_depth = np.full_like(grid_x, float(depth), dtype=float)
    rgba = colormaps["inferno"](array)
    ax.plot_surface(
        grid_depth,
        grid_x,
        grid_y,
        facecolors=rgba,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0,
    )
    ax.plot(
        [depth] * 5,
        [x1, x2, x2, x1, x1],
        [image_height - y1, image_height - y1, image_height - y2, image_height - y2, image_height - y1],
        color=edgecolor,
        linewidth=linewidth,
    )


def plot_multiscale_fusion_overview(
    trace: dict[str, Any],
    config: LoopPanelConfig,
) -> plt.Figure:
    """Show every local heatmap at its true position in global coordinates."""
    count = loop_count(trace, config)
    image_width, image_height = trace["image"].size
    stages = list(trace.get("stages") or [])[:count]
    source_layers: list[tuple[np.ndarray, tuple[int, int, int, int]]] = [
        (sanitize_heatmap(trace["initial_heatmap"]), (0, 0, image_width, image_height))
    ]
    source_layers.extend(
        (sanitize_heatmap(stage["local_heatmap"]), bbox_to_int_tuple(stage["crop_bbox"]))
        for stage in stages
        if stage.get("local_heatmap") is not None
    )
    final_heatmap = final_heatmap_for_trace(trace, stages)

    source_gap = max(image_width, image_height) * 0.34
    result_gap = source_gap * 1.75
    source_depths = [index * source_gap for index in range(len(source_layers))]
    result_depth = (source_depths[-1] if source_depths else 0.0) + result_gap

    fig = plt.figure(figsize=config.fusion_overview_figsize, dpi=int(config.display_dpi))
    ax = fig.add_subplot(111, projection="3d")
    for depth, (heatmap, bbox) in zip(source_depths, source_layers):
        _draw_global_coordinate_frame(ax, depth, (image_width, image_height))
        _draw_heatmap_surface(
            ax,
            heatmap,
            depth=depth,
            image_bbox=bbox,
            image_height=image_height,
            texture_side=config.fusion_surface_resolution,
            edgecolor=WHITE,
            linewidth=0.72,
        )

    _draw_heatmap_surface(
        ax,
        final_heatmap,
        depth=result_depth,
        image_bbox=(0, 0, image_width, image_height),
        image_height=image_height,
        texture_side=config.fusion_surface_resolution,
        edgecolor=RED,
        linewidth=1.05,
    )
    ax.set_xlim(-source_gap * 0.18, result_depth + source_gap * 0.18)
    ax.set_ylim(0, image_width)
    ax.set_zlim(0, image_height)
    ax.set_box_aspect((max(result_depth, 1.0), image_width, image_height))
    ax.view_init(elev=float(config.fusion_overview_elev), azim=float(config.fusion_overview_azim))
    ax.set_axis_off()
    ax.set_proj_type("persp", focal_length=0.9)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig


def display_multiscale_fusion_overview(
    trace: dict[str, Any],
    config: LoopPanelConfig,
) -> None:
    fig = plot_multiscale_fusion_overview(trace, config)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if config.save:
        save_dir = Path(config.save_dir)
        if not save_dir.is_absolute():
            save_dir = REPO_ROOT / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(trace["row"].get("id", "sample"))
        out_path = save_dir / f"{sample_id}_multiscale_fusion_overview.png"
        fig.savefig(
            out_path,
            dpi=max(420, int(config.dpi), int(config.fusion_overview_resolution)),
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"saved: {out_path}")
    display(fig)
    plt.close(fig)


def _heatmap_data_uri(heatmap: Any, texture_side: int | None) -> str:
    array = heatmap_for_display(heatmap)
    height, width = array.shape
    if texture_side is None:
        out_width, out_height = width, height
    else:
        texture_side = max(8, int(texture_side))
        scale = float(texture_side) / max(height, width)
        out_width = max(2, int(round(width * scale)))
        out_height = max(2, int(round(height * scale)))
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    if (out_width, out_height) != (width, height):
        image = image.resize((out_width, out_height), Image.Resampling.BICUBIC)
    values = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    rgba = np.asarray(colormaps["inferno"](values) * 255.0, dtype=np.uint8)
    png = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    png.save(buffer, format="PNG", compress_level=1)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_data_uri(image: Image.Image, texture_side: int | None = None) -> str:
    image = image.convert("RGB")
    width, height = image.size
    if texture_side is not None:
        texture_side = max(8, int(texture_side))
        scale = min(1.0, float(texture_side) / max(width, height))
        out_size = (
            max(2, int(round(width * scale))),
            max(2, int(round(height * scale))),
        )
        if out_size != image.size:
            image = image.resize(out_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _html_stage_boxes(trace: dict[str, Any], config: LoopPanelConfig) -> list[dict[str, Any]]:
    count = loop_count(trace, config)
    boxes: list[dict[str, Any]] = []
    for stage in list(trace.get("stages") or [])[:count]:
        if not stage.get("crop_bbox"):
            continue
        stage_id = stage.get("stage")
        label = f"S{stage_id}" if stage_id is not None else "stage"
        target = stage.get("target_label")
        if target:
            label = f"{label} {target}"
        boxes.append(
            {
                "bbox": list(bbox_to_int_tuple(stage["crop_bbox"])),
                "label": label,
                "final": False,
            }
        )
    if boxes:
        final_bbox = trace.get("final_bbox")
        final_box = bbox_to_int_tuple(final_bbox) if final_bbox else tuple(boxes[-1]["bbox"])
        for box in reversed(boxes):
            if tuple(box["bbox"]) == final_box:
                box["label"] = f"{box['label']} / final"
                box["final"] = True
                break
        else:
            boxes.append({"bbox": list(final_box), "label": "final", "final": True})
    return boxes


def _same_component(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    for key in ("bbox", "image_bbox"):
        if a.get(key) and b.get(key):
            return bbox_to_int_tuple(a[key]) == bbox_to_int_tuple(b[key])
    return False


def _html_topmass(config: LoopPanelConfig) -> float:
    if config.topmass is not None:
        return float(config.topmass)
    return float(MODEL_DEFAULTS[str(config.model)]["topmass"])


def _html_stage_details(trace: dict[str, Any], config: LoopPanelConfig) -> list[dict[str, Any]]:
    count = loop_count(trace, config)
    topmass = _html_topmass(config)
    image_width, image_height = trace["image"].size
    details: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(list(trace.get("stages") or [])[:count], start=1):
        if stage.get("pre_heatmap") is None:
            continue
        heatmap = sanitize_heatmap(stage["pre_heatmap"])
        heatmap_height, heatmap_width = heatmap.shape
        selection = stage.get("selection") if isinstance(stage.get("selection"), dict) else {}
        selected = selection.get("selected_component") if isinstance(selection, dict) else None
        mask_info = topmass_mask(heatmap, topmass)
        components = connected_components(mask_info["mask"], heatmap)

        boxes: list[dict[str, Any]] = []
        for component_index, component in enumerate(components, start=1):
            if not isinstance(component, dict):
                continue
            is_selected = _same_component(component, selected)
            if selected is None and component_index == 1:
                is_selected = True
            if not component.get("bbox"):
                continue
            bbox = bbox_to_int_tuple(component["bbox"])
            boxes.append(
                {
                    "bbox": list(bbox),
                    "label": "selected" if is_selected else f"candidate {component_index}",
                    "selected": bool(is_selected),
                }
            )
        if not boxes:
            continue
        if not any(box.get("selected") for box in boxes):
            boxes[0]["label"] = "selected"
            boxes[0]["selected"] = True
        detail = {
            "stage": int(stage.get("stage") or stage_index),
            "label": stage_label(stage),
            "componentCount": int(len(components)),
            "heatmapWidth": int(heatmap_width),
            "heatmapHeight": int(heatmap_height),
            "heatmapFrameBBox": list(
                bbox_to_int_tuple(stage.get("search_frame_bbox") or (0, 0, image_width, image_height))
            ),
            "topmass": float(topmass),
            "boxes": boxes,
        }
        detail["heatUri"] = _heatmap_data_uri(heatmap, None)
        details.append(detail)
    return details


def stage_boxes_preview_image(
    trace: dict[str, Any],
    config: LoopPanelConfig,
    *,
    max_side: int = 1024,
) -> Image.Image:
    """Return a notebook-only low-resolution full-image stage-box preview."""
    image = trace["image"].convert("RGB")
    width, height = image.size
    max_side = max(64, int(max_side))
    scale = min(1.0, float(max_side) / max(width, height))
    preview_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    preview = image.resize(preview_size, Image.Resampling.LANCZOS) if scale < 1 else image.copy()
    draw = ImageDraw.Draw(preview)
    line_width = max(2, int(round(max(preview_size) / 420)))
    dash = max(8, int(round(max(preview_size) / 80)))
    gap = max(5, int(round(dash * 0.65)))
    for index, box in enumerate(_html_stage_boxes(trace, config)):
        x1, y1, x2, y2 = [int(round(float(value) * scale)) for value in box["bbox"]]
        color = SELECTED_BOX_COLOR if box.get("final") else STAGE_BOX_COLORS[index % len(STAGE_BOX_COLORS)]
        if box.get("final"):
            for offset in range(line_width):
                draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        else:
            draw_dashed_rectangle(
                draw,
                (x1, y1, x2, y2),
                color=color,
                width=line_width,
                dash=dash,
                gap=gap,
            )
    return preview


def stage_boxes_legend_image(
    trace: dict[str, Any],
    config: LoopPanelConfig,
    *,
    max_width: int = 1600,
) -> Image.Image:
    boxes = _html_stage_boxes(trace, config)
    max_width = max(640, int(max_width))
    font_size = max(22, int(round(max_width / 58)))
    font = times_new_roman_font(font_size)
    row_height = max(42, int(round(font_size * 1.55)))
    margin = max(24, int(round(font_size * 1.0)))
    swatch_width = max(74, int(round(font_size * 3.2)))
    height = max(row_height + margin * 2, margin * 2 + row_height * len(boxes))
    legend = Image.new("RGB", (max_width, height), WHITE)
    draw = ImageDraw.Draw(legend)
    for index, box in enumerate(boxes):
        y = margin + index * row_height + row_height // 2
        color = SELECTED_BOX_COLOR if box.get("final") else STAGE_BOX_COLORS[index % len(STAGE_BOX_COLORS)]
        line_width = max(5, int(round(font_size / 5))) if box.get("final") else max(4, int(round(font_size / 6)))
        x1 = margin
        x2 = margin + swatch_width
        if box.get("final"):
            draw.line((x1, y, x2, y), fill=color, width=line_width)
        else:
            dash = max(18, int(round(font_size * 0.9)))
            gap = max(10, int(round(font_size * 0.5)))
            for dash_x in range(x1, x2, dash + gap):
                draw.line((dash_x, y, min(dash_x + dash, x2), y), fill=color, width=line_width)
        label = str(box.get("label") or ("final" if box.get("final") else f"stage {index + 1}"))
        text_box = draw.textbbox((0, 0), label, font=font)
        text_height = text_box[3] - text_box[1]
        draw.text((x2 + 18, y - text_height // 2), label, fill=DARK, font=font)
    return legend


def display_stage_boxes_preview(
    trace: dict[str, Any],
    config: LoopPanelConfig,
    *,
    max_side: int = 1024,
    legend_width: int | None = None,
) -> tuple[Image.Image, Image.Image]:
    preview = stage_boxes_preview_image(trace, config, max_side=max_side)
    legend = stage_boxes_legend_image(
        trace,
        config,
        max_width=max(1600, max_side) if legend_width is None else int(legend_width),
    )
    display(preview)
    display(legend)
    return preview, legend


def display_trace_text_summary(trace: dict[str, Any]) -> None:
    row = trace.get("row") or {}
    sample_id = row.get("id", "sample")
    question = trace.get("question") or row.get("question") or ""
    answer_text = trace.get("answer_text") or ""
    original_answer = display_answer(trace.get("original_answer"), limit=220)
    crop_answer = display_answer(trace.get("crop_answer"), limit=220)
    original_source = trace.get("original_answer_source") or "unknown"
    crop_source = trace.get("crop_answer_source") or "unknown"
    original_score = trace.get("original_score")
    crop_score = trace.get("crop_score")

    def score_text(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    rows = [
        ("Sample", str(sample_id)),
        ("Question", str(question)),
    ]
    if answer_text:
        rows.append(("Ground truth", str(answer_text)))
    rows.extend(
        [
            (
                "NoCrop answer",
                f"{html.escape(original_answer)} "
                f"<span class=\"muted\">source={html.escape(str(original_source))}; "
                f"score={html.escape(score_text(original_score))}</span>",
            ),
            (
                "PHF-Crop answer",
                f"{html.escape(crop_answer)} "
                f"<span class=\"muted\">source={html.escape(str(crop_source))}; "
                f"score={html.escape(score_text(crop_score))}</span>",
            ),
        ]
    )
    body = "\n".join(
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{value if label.endswith('answer') else html.escape(value)}</td>"
        "</tr>"
        for label, value in rows
    )
    display(
        HTML(
            """
<style>
.phf-summary { font-family: "Times New Roman", Tinos, serif; max-width: 980px; margin: 8px 0 14px; border-collapse: collapse; }
.phf-summary th { width: 150px; text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid #e5e7eb; color: #111827; font-weight: 700; }
.phf-summary td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; color: #111827; line-height: 1.35; }
.phf-summary .muted { color: #6b7280; margin-left: 8px; white-space: nowrap; font-size: 0.92em; }
</style>
<table class="phf-summary">
"""
            + body
            + "\n</table>"
        )
    )


def _html_text_summary(trace: dict[str, Any]) -> dict[str, Any]:
    row = trace.get("row") or {}
    sample_id = row.get("id", "sample")
    question = trace.get("question") or row.get("question") or ""
    answer_text = trace.get("answer_text") or ""

    def score_text(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    nocrop = {
        "answer": display_answer(trace.get("original_answer"), limit=1000),
        "source": str(trace.get("original_answer_source") or "unknown"),
        "score": score_text(trace.get("original_score")),
    }
    current = {
        "answer": display_answer(trace.get("crop_answer"), limit=1000),
        "source": str(trace.get("crop_answer_source") or "unknown"),
        "score": score_text(trace.get("crop_score")),
    }
    return {
        "sample": str(sample_id),
        "question": str(question),
        "groundTruth": str(answer_text) if answer_text else "",
        "nocrop": nocrop,
        "current": current,
        "phf": current,
    }


def _html_scene_layers(trace: dict[str, Any], config: LoopPanelConfig) -> list[dict[str, Any]]:
    count = loop_count(trace, config)
    image_width, image_height = trace["image"].size
    stages = list(trace.get("stages") or [])[:count]
    source_layers: list[tuple[Any, tuple[int, int, int, int]]] = [
        (
            sanitize_heatmap(trace["initial_heatmap"]),
            (0, 0, image_width, image_height),
        )
    ]
    source_layers.extend(
        (
            sanitize_heatmap(stage["local_heatmap"]),
            bbox_to_int_tuple(stage["crop_bbox"]),
        )
        for stage in stages
        if stage.get("local_heatmap") is not None
    )
    final_heatmap = final_heatmap_for_trace(trace, stages)

    depth_gap = 132.0
    result_gap = 230.0
    layers: list[dict[str, Any]] = []
    for index, (heatmap, bbox) in enumerate(source_layers):
        layers.append(
            {
                "bbox": list(bbox),
                "depth": float(index * depth_gap),
                "final": False,
                "heatUri": _heatmap_data_uri(heatmap, config.web_texture_resolution),
            }
        )
    result_depth = (len(source_layers) - 1) * depth_gap + result_gap
    layers.append(
        {
            "bbox": [0, 0, image_width, image_height],
            "depth": float(result_depth),
            "final": True,
            "heatUri": _heatmap_data_uri(final_heatmap, config.web_texture_resolution),
        }
    )
    return layers


def build_html_multiscale_fusion_view(trace: dict[str, Any], config: LoopPanelConfig) -> str:
    """Build a self-contained interactive WebGL multiscale fusion view."""
    image_width, image_height = trace["image"].size
    layers = _html_scene_layers(trace, config)
    payload = {
        "imageWidth": int(image_width),
        "imageHeight": int(image_height),
        "originalUri": _image_data_uri(trace["image"], None),
        "boxes": _html_stage_boxes(trace, config),
        "stageDetails": _html_stage_details(trace, config),
        "text": _html_text_summary(trace),
        "layers": layers,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    template_path = Path(__file__).with_name("fusion_web_template.html")
    if not template_path.exists():
        raise FileNotFoundError(f"Missing WebGL template: {template_path}")
    return template_path.read_text(encoding="utf-8").replace("__PHF_PAYLOAD__", payload_json)


def write_html_multiscale_fusion_view(
    trace: dict[str, Any],
    config: LoopPanelConfig,
    out_path: Path | str | None = None,
) -> Path:
    if out_path is None:
        save_dir = Path(config.save_dir)
        if not save_dir.is_absolute():
            save_dir = REPO_ROOT / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(trace["row"].get("id", "sample"))
        out_path = save_dir / f"{sample_id}_fusion_web.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html_multiscale_fusion_view(trace, config), encoding="utf-8")
    return out_path


def display_html_multiscale_fusion_view(
    trace: dict[str, Any],
    config: LoopPanelConfig,
) -> Path:
    out_path = write_html_multiscale_fusion_view(trace, config)
    print(f"open in browser: {out_path}")
    try:
        relative_path = out_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        browser_path = f"/files/{relative_path}"
    except ValueError:
        browser_path = out_path.as_uri()
    display(HTML(f'<a href="{browser_path}" target="_blank">Open interactive fusion view</a>'))
    return out_path


def plot_loop_panel(
    trace: dict[str, Any],
    context: Any,
    config: LoopPanelConfig,
    loop_index: int,
) -> plt.Figure:
    current_image, current_heatmap, current_label = current_image_and_heatmap(trace, loop_index)
    fused_heatmap = fused_heatmap_for_loop(trace, loop_index)
    if fused_heatmap is None:
        fig, axes = plt.subplots(1, 3, figsize=config.figsize_3)
        show_image(
            axes[0],
            current_image,
            f"Loop {loop_index}: current image\n{image_size_text(current_image)}"
            if config.show_text
            else None,
        )
        show_heatmap_values(
            axes[1],
            current_heatmap,
            f"current heatmap values ({heatmap_size_text(current_heatmap)})\n{current_label}"
            if config.show_text
            else None,
        )
        draw_topmass_heatmap(
            axes[2],
            trace,
            context,
            loop_index,
            max_components=config.max_components,
            show_text=config.show_text,
        )
    else:
        fig, axes = plt.subplots(1, 4, figsize=config.figsize_4)
        show_image(
            axes[0],
            current_image,
            f"Loop {loop_index}: current crop\n{image_size_text(current_image)}"
            if config.show_text
            else None,
        )
        show_heatmap_values(
            axes[1],
            current_heatmap,
            f"current heatmap values ({heatmap_size_text(current_heatmap)})\n{current_label}"
            if config.show_text
            else None,
        )
        show_heatmap_values(
            axes[2],
            fused_heatmap,
            f"fused heatmap values ({heatmap_size_text(fused_heatmap)})"
            if config.show_text
            else None,
        )
        draw_topmass_heatmap(
            axes[3],
            trace,
            context,
            loop_index,
            max_components=config.max_components,
            show_text=config.show_text,
        )

    stop = trace.get("adaptive_stop") or {}
    stop_text = ""
    if stop:
        stop_text = (
            f" | adaptive stop selects S{stop.get('selected_stage')} "
            f"({stop.get('selected_target_label')}, "
            f"short-ratio={float(stop.get('selected_component_short_ratio', 0.0)):.2f})"
        )
    if config.show_text:
        fig.suptitle(
            f"PHF-Crop loop {loop_index}: heatmap -> fusion -> topmass component crop{stop_text}",
            fontsize=12,
            fontweight="bold",
            y=1.03,
        )
    return fig


def display_loop_panels(trace: dict[str, Any], context: Any, config: LoopPanelConfig) -> None:
    count = loop_count(trace, config)
    print(
        f"displaying {count} loop panel(s); "
        f"sample={trace['row'].get('id')}, model={context.config.model}, task={context.config.task}"
    )
    for loop_index in range(count):
        current_image, current_heatmap, _ = current_image_and_heatmap(trace, loop_index)
        fused_heatmap = fused_heatmap_for_loop(trace, loop_index)
        selection_stage = selection_stage_for_loop(trace, loop_index)
        crop_bbox = bbox_to_int_tuple(selection_stage["crop_bbox"])
        crop_w = max(1, crop_bbox[2] - crop_bbox[0])
        crop_h = max(1, crop_bbox[3] - crop_bbox[1])
        fused_text = "none" if fused_heatmap is None else heatmap_size_text(fused_heatmap)
        print(
            f"loop {loop_index}: current={image_size_text(current_image)}, "
            f"current_heatmap={heatmap_size_text(current_heatmap)}, "
            f"fused_heatmap={fused_text}, "
            f"next_crop={crop_w} x {crop_h} px ({stage_label(selection_stage)})"
        )
        fig = plot_single_image_panel(
            current_image,
            f"Loop {loop_index}: current image\n{image_size_text(current_image)}"
            if config.show_text
            else None,
        )
        save_or_display(fig, trace, config, loop_index, "current_image")

        fig = plot_single_original_next_crop_panel(
            trace,
            loop_index,
            f"Loop {loop_index}: original image with next crop\n{image_size_text(trace['image'])}"
            if config.show_text
            else None,
        )
        save_or_display(fig, trace, config, loop_index, "original_next_crop")

        fig = plot_single_heatmap_panel(
            current_heatmap,
            f"Loop {loop_index}: current heatmap values ({heatmap_size_text(current_heatmap)})"
            if config.show_text
            else None,
        )
        save_or_display(fig, trace, config, loop_index, "current_heatmap")

        if fused_heatmap is not None:
            fig = plot_single_heatmap_panel(
                fused_heatmap,
                f"Loop {loop_index}: fused heatmap values ({heatmap_size_text(fused_heatmap)})"
                if config.show_text
                else None,
            )
            save_or_display(fig, trace, config, loop_index, "fused_heatmap")

        fig = plot_single_top1_panel(trace, context, config, loop_index)
        save_or_display(fig, trace, config, loop_index, "top1_heatmap")


def release_adapter(adapter: Any) -> None:
    try:
        import torch

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
