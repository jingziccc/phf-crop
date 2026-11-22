#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from top1_hotspot.config import (  # noqa: E402
    ADAPTIVE_STOP_RATIO,
    COMMON_MIN_CROP_SIDE,
    FROZEN_CONFIG,
    QWEN_VISUAL_TOKENS,
)
from top1_hotspot.geometry import build_stage_targets  # noqa: E402
from top1_hotspot.adaptive_stop import component_short_ratio  # noqa: E402
from top1_hotspot.pipeline import compact_stage, run_top1_hotspot_chain  # noqa: E402


class HeatmapCounter:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, *_args: object) -> np.ndarray:
        self.calls += 1
        return self.value.copy()


def adapter_with(counter: HeatmapCounter) -> SimpleNamespace:
    return SimpleNamespace(
        min_crop_side=COMMON_MIN_CROP_SIDE,
        general_prompt="general",
        model=None,
        processor=None,
        heatmap_fn=counter,
    )


def test_exact_greater_equal_stop() -> None:
    image = Image.new("RGB", (100, 100))
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[2:7, 2:7] = 1.0
    counter = HeatmapCounter(heatmap)
    stages = run_top1_hotspot_chain(
        image=image,
        adapter=adapter_with(counter),
        attention_prompt="question",
        initial_heatmap=heatmap,
        targets=[
            {"kind": "area_ratio", "ratio": 0.25, "side": 50},
            {"kind": "min_crop_side", "ratio": None, "side": 25},
        ],
        topmass=1.0,
        adaptive_stop_hotspot_ratio=1.0,
    )
    assert len(stages) == 1
    stage = stages[0]
    assert stage["adaptive_stop_component_short_ratio"] == 1.0
    assert stage["stop_reason"] == "hotspot_fills_candidate"
    assert stage["final_bbox_semantics"] == "current_centered_square_crop"
    assert counter.calls == 0


def test_invalid_component_cannot_stop() -> None:
    image = Image.new("RGB", (100, 100))
    heatmap = np.zeros((10, 10), dtype=np.float32)
    counter = HeatmapCounter(heatmap)
    stages = run_top1_hotspot_chain(
        image=image,
        adapter=adapter_with(counter),
        attention_prompt="question",
        initial_heatmap=heatmap,
        targets=[
            {"kind": "area_ratio", "ratio": 0.25, "side": 50},
            {"kind": "min_crop_side", "ratio": None, "side": 25},
        ],
        topmass=0.5,
        adaptive_stop_hotspot_ratio=1.0,
    )
    assert len(stages) == 2
    assert stages[0]["selection"]["selected_component"] is None
    assert not stages[0].get("adaptive_stop_triggered", False)
    assert stages[-1]["stop_reason"] == "min_crop_side"
    assert counter.calls == 1


def test_compact_selection_contract() -> None:
    image = Image.new("RGB", (100, 100))
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[1:3, 1:3] = 2.0
    heatmap[7:9, 7:9] = 1.0
    counter = HeatmapCounter(heatmap)
    stages = run_top1_hotspot_chain(
        image=image,
        adapter=adapter_with(counter),
        attention_prompt="question",
        initial_heatmap=heatmap,
        targets=[{"kind": "min_crop_side", "ratio": None, "side": 25}],
        topmass=1.0,
        adaptive_stop_hotspot_ratio=1.0,
    )
    compact = compact_stage(stages[0])
    selection = compact["selection"]
    assert "components" not in selection
    assert selection["component_count"] == 2
    assert selection["selected_component"] is not None
    assert selection["runner_up_component"] is not None
    assert "pre_heatmap" not in compact
    assert len(json.dumps(compact, ensure_ascii=False)) < 8000


def test_rectangular_candidate_uses_candidate_short_side() -> None:
    stage = {
        "side": 50,
        "crop_bbox": [10, 10, 60, 40],
        "selection": {
            "selected_component": {"image_bbox": [10, 10, 60, 40]},
        },
    }
    assert component_short_ratio(stage) == 1.0


def test_frozen_constants() -> None:
    assert ADAPTIVE_STOP_RATIO == 1.0
    assert FROZEN_CONFIG["shared"]["adaptive_stop_operator"] == ">="
    assert FROZEN_CONFIG["shared"]["final_bbox_semantics"] == "current_centered_square_crop"
    assert COMMON_MIN_CROP_SIDE == 336
    assert QWEN_VISUAL_TOKENS == 256
    assert FROZEN_CONFIG["models"]["qwen2_5"]["max_pixels"] == 200704
    assert build_stage_targets((1920, 1080), min_crop_side=336)[-1]["kind"] == "min_crop_side"


def main() -> None:
    tests = [
        test_exact_greater_equal_stop,
        test_invalid_component_cannot_stop,
        test_compact_selection_contract,
        test_rectangular_candidate_uses_candidate_short_side,
        test_frozen_constants,
    ]
    for test in tests:
        test()
        print(f"[pass] {test.__name__}")
    print(f"[ok] frozen PHF contract validated ({len(tests)} checks)")


if __name__ == "__main__":
    main()
