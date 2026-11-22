from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SUPPORTED_MODELS = ("llava", "qwen2_5", "blip")

METHOD_NAME = "phf_crop"
METHOD_DISPLAY_NAME = "PHF-Crop"
METHOD_CN_NAME = "渐进式热点融合裁剪"
METHOD_FULL_NAME = "Progressive Hotspot Fusion Crop"

FROZEN_CONFIG_PATH = Path(__file__).with_name("frozen_config.json")


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _load_frozen_config() -> dict[str, Any]:
    value = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if value.get("version") != "phf_crop_final_20260713":
        raise RuntimeError(f"unexpected frozen config version: {value.get('version')!r}")
    return value


FROZEN_CONFIG = _load_frozen_config()
COMMON_MIN_CROP_SIDE = int(FROZEN_CONFIG["shared"]["min_crop_side"])
ADAPTIVE_STOP_RATIO = float(FROZEN_CONFIG["shared"]["adaptive_stop_ratio"])
QWEN_VISUAL_TOKENS = int(FROZEN_CONFIG["models"]["qwen2_5"]["visual_tokens"])


def frozen_config_sha256() -> str:
    return hashlib.sha256(FROZEN_CONFIG_PATH.read_bytes()).hexdigest()


def _method_name(model: str, topmass: float) -> str:
    topmass_tag = float_tag(topmass)
    if model == "blip":
        return (
            "phf_crop_blip_rawsub_alpha1p0"
            f"_tm{topmass_tag}_minside{COMMON_MIN_CROP_SIDE}"
        )
    model_tag = "qwen" if model == "qwen2_5" else model
    return (
        f"phf_crop_{model_tag}_direct_tm{topmass_tag}"
        f"_minside{COMMON_MIN_CROP_SIDE}"
    )


MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    model: {
        "topmass": float(settings["topmass"]),
        "min_crop_side": COMMON_MIN_CROP_SIDE,
        "vit_input_size": COMMON_MIN_CROP_SIDE,
        "blip_bg_sub_alpha": float(settings.get("background_subtraction_alpha", 1.0)),
        "method_name": _method_name(model, float(settings["topmass"])),
        "topmass_locked": bool(settings["topmass_locked"]),
    }
    for model, settings in FROZEN_CONFIG["models"].items()
}


def model_defaults(model: str) -> dict[str, Any]:
    try:
        return dict(MODEL_DEFAULTS[str(model)])
    except KeyError as exc:
        raise ValueError(f"unsupported PHF-Crop model: {model!r}") from exc


def default_method_name_for_params(
    *,
    model: str,
    topmass: float,
    min_crop_side: int,
    blip_bg_sub_alpha: float = 1.0,
) -> str:
    defaults = model_defaults(model)
    default_alpha = float(defaults.get("blip_bg_sub_alpha", 1.0))
    if (
        abs(float(topmass) - float(defaults["topmass"])) < 1e-12
        and int(min_crop_side) == int(defaults["min_crop_side"])
        and abs(float(blip_bg_sub_alpha) - default_alpha) < 1e-12
    ):
        return str(defaults["method_name"])

    topmass_tag = float_tag(float(topmass))
    if model == "blip":
        alpha_tag = float_tag(float(blip_bg_sub_alpha))
        return (
            f"phf_crop_blip_rawsub_no_norm_alpha{alpha_tag}"
            f"_topmass{topmass_tag}_minside{int(min_crop_side)}"
        )
    model_tag = "qwen" if model == "qwen2_5" else str(model)
    return f"phf_crop_{model_tag}_direct_tm{topmass_tag}_minside{int(min_crop_side)}"


def resolve_model_params(
    *,
    model: str,
    topmass: float | None = None,
    min_crop_side: int | None = None,
    blip_bg_sub_alpha: float | None = None,
    method_name: str | None = None,
) -> dict[str, Any]:
    defaults = model_defaults(model)
    resolved_topmass = float(defaults["topmass"] if topmass is None else topmass)
    resolved_min_side = int(
        defaults["min_crop_side"] if min_crop_side is None else min_crop_side
    )
    resolved_blip_alpha = float(
        defaults.get("blip_bg_sub_alpha", 1.0)
        if blip_bg_sub_alpha is None
        else blip_bg_sub_alpha
    )
    resolved_method = method_name or default_method_name_for_params(
        model=model,
        topmass=resolved_topmass,
        min_crop_side=resolved_min_side,
        blip_bg_sub_alpha=resolved_blip_alpha,
    )
    return {
        "topmass": resolved_topmass,
        "min_crop_side": resolved_min_side,
        "vit_input_size": resolved_min_side,
        "blip_bg_sub_alpha": resolved_blip_alpha,
        "method_name": resolved_method,
    }
