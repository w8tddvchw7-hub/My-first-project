"""Step 2: turn the analysis into Lightroom slider values.

Every number here lives in DEFAULT_CONFIG so it can be calibrated without
touching code: `python -m idan_auto config > my_config.json`, edit, then
`run --config my_config.json`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

DEFAULT_CONFIG: dict = {
    # Target *linear* luminance of the car body, per car tone. This is what
    # normalises a whole set: ten cars shot at different times all land on
    # the same brightness for their colour.
    "exposure_targets": {"black": 0.03, "dark": 0.06, "silver": 0.32,
                         "white": 0.55, "red": 0.12, "color": 0.14},
    # Exposure is a blend of "car on target" and "frame on target", so a black
    # car in a bright hall is lifted a little, not until the walls blow out.
    "exposure_car_weight": 0.85,
    "frame_target": 0.35,        # target median linear luminance of the frame
    "exposure_bias": 0.0,        # global EV shift (calibrate vs. Lightroom rendering)
    "exposure_damping": 0.85,    # 1.0 = hit the target exactly
    "exposure_limit": 2.0,       # max EV correction either way

    "wb_strength": 0.55,         # how much of the measured colour cast to remove
    "wb_deadband": 0.04,         # ignore casts smaller than this (ln R/B)
    "mired_per_ln": 180.0,       # ln(R/B) -> mired shift
    "tint_per_ln": 120.0,        # ln(G/sqrt(RB)) -> Lightroom tint

    "base": {
        "Contrast2012": 5, "Highlights2012": -40, "Shadows2012": 25,
        "Whites2012": 10, "Blacks2012": -10, "Texture": 10, "Clarity2012": 8,
        "Dehaze": 4, "Vibrance": 12, "Saturation": 0, "Sharpness": 40,
        "LuminanceSmoothing": 0, "ColorNoiseReduction": 25,
    },
    # Added on top of "base" according to the car's tone.
    "by_tone": {
        "black": {"Shadows2012": 20, "Blacks2012": 8, "Clarity2012": -3,
                  "Texture": 5, "Highlights2012": -10},
        "dark": {"Shadows2012": 10, "Blacks2012": 4},
        "silver": {"Highlights2012": -20, "Whites2012": -8, "Clarity2012": 4, "Texture": 5},
        "white": {"Highlights2012": -30, "Whites2012": -20, "Shadows2012": -5,
                  "Clarity2012": 5, "Dehaze": 3, "Vibrance": -4,
                  "SaturationAdjustmentYellow": -15, "SaturationAdjustmentOrange": -5},
        "red": {"SaturationAdjustmentRed": -12, "LuminanceAdjustmentRed": -8,
                "SaturationAdjustmentOrange": -5, "Highlights2012": -10, "Vibrance": -4},
        "color": {"Vibrance": 4},
    },
    "histogram": {
        "flat_std": 0.16, "flat": {"Contrast2012": 10, "Dehaze": 4},
        "clip_hi": 0.03, "clipped_highlights": {"Highlights2012": -15, "Whites2012": -10},
        "clip_lo": 0.05, "crushed_shadows": {"Shadows2012": 10, "Blacks2012": 8},
    },
    "light": {
        "low_light": {"Shadows2012": 8},
        "artificial": {"SaturationAdjustmentYellow": -10},
        "nr_per_stop": 9,        # LuminanceSmoothing per stop above ISO 100
        "nr_max": 50,
    },
    # Interior / detail shots get their own rules instead of the car-tone ones.
    "interior": {
        "deltas": {"Highlights2012": -35, "Shadows2012": 20, "Whites2012": -10,
                   "Blacks2012": 5, "Clarity2012": -3, "Vibrance": -4},
        "frame_target": 0.05,    # cabins are meant to look dark and rich
        "exposure_limit": 1.0,
        "wb_strength": 0.4,      # screens and ambient LEDs fool the WB estimate
    },
    # Applied to every photo (lens corrections etc.).
    "fixed": {"SharpenRadius": 1.0, "SharpenDetail": 25, "SharpenEdgeMasking": 50,
              "LensProfileEnable": 1, "AutoLateralCA": 1},
}

RANGES = {
    "Exposure2012": (-5, 5), "Temperature": (2000, 50000), "Tint": (-150, 150),
    "IncrementalTemperature": (-100, 100), "IncrementalTint": (-100, 100),
    "Sharpness": (0, 150), "LuminanceSmoothing": (0, 100), "ColorNoiseReduction": (0, 100),
}


def load_config(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        _deep_update(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _add(s: dict, delta: dict) -> None:
    for k, v in delta.items():
        s[k] = s.get(k, 0) + v


def clamp_and_round(s: dict) -> dict:
    out = {}
    for k, v in s.items():
        lo, hi = RANGES.get(k, (-100, 100))
        if k in ("SharpenRadius",):
            out[k] = round(float(v), 1)
        elif k == "Exposure2012":
            out[k] = round(min(hi, max(lo, float(v))), 2)
        elif k in ("LensProfileEnable", "AutoLateralCA"):
            out[k] = int(v)
        else:
            out[k] = int(round(min(hi, max(lo, float(v)))))
    return out


def compute_settings(a: dict, cfg: dict) -> tuple[dict, dict]:
    """Return (raw slider values, meta describing the decisions)."""
    s: dict = {}
    reasons: list[str] = []
    _add(s, cfg["base"])

    tone = a["car_tone"]
    interior = a.get("scene") == "interior"
    icfg = cfg["interior"]
    if interior:
        _add(s, icfg["deltas"])
        reasons.append("interior / detail shot")
    else:
        _add(s, cfg["by_tone"].get(tone, {}))
        reasons.append(f"car tone: {tone} ({a['car_color']})")

    h = cfg["histogram"]
    if a["frame_std"] < h["flat_std"]:
        _add(s, h["flat"]); reasons.append("flat histogram -> more contrast")
    if a["clip_hi"] > h["clip_hi"]:
        _add(s, h["clipped_highlights"]); reasons.append("clipped highlights")
    if a["clip_lo"] > h["clip_lo"]:
        _add(s, h["crushed_shadows"]); reasons.append("crushed shadows")

    light = cfg["light"]
    if a["low_light"] or a["evening"]:
        _add(s, light["low_light"]); reasons.append("evening / low light")
    if a["artificial_light"]:
        _add(s, light["artificial"]); reasons.append("artificial (hall) light")
    iso = a.get("iso") or 100
    if iso > 100:
        stops = math.log2(iso / 100)
        s["LuminanceSmoothing"] = max(s.get("LuminanceSmoothing", 0),
                                      min(light["nr_max"], light["nr_per_stop"] * stops))
        if iso >= 3200:
            s["ColorNoiseReduction"] = s.get("ColorNoiseReduction", 25) + 10
            s["Sharpness"] = s.get("Sharpness", 40) - 10
            reasons.append(f"ISO {int(iso)} -> noise reduction")

    # Exposure: bring the car body to the target brightness for its tone.
    target = cfg["exposure_targets"].get(tone, 0.14)
    ev_car = math.log2(target / max(a["car_lin"], 1e-4))
    frame_target = icfg["frame_target"] if interior else cfg["frame_target"]
    ev_frame = math.log2(frame_target / max(a.get("frame_lin_med", frame_target), 1e-4))
    w = 0.0 if interior else cfg["exposure_car_weight"]
    ev = (w * ev_car + (1 - w) * ev_frame) * cfg["exposure_damping"] + cfg["exposure_bias"]
    limit = icfg["exposure_limit"] if interior else cfg["exposure_limit"]
    ev = max(-limit, min(limit, ev))
    if ev > 0 and a["car_clip_hi"] > 0.02 and not interior:
        ev = min(ev, 0.1)
        reasons.append("paint already clipping -> no brightening")
    s["Exposure2012"] = ev

    # White balance: remove the measured cast.
    meta = {"reasons": reasons, "wb_ln_rb": 0.0, "wb_ln_g": 0.0}
    rb, gm = a["cast_rb"], a["cast_g"]
    if abs(rb) > cfg["wb_deadband"] or abs(gm) > cfg["wb_deadband"]:
        strength = icfg["wb_strength"] if interior else cfg["wb_strength"]
        rb_c = rb * strength
        gm_c = gm * strength
        meta.update(wb_ln_rb=rb_c, wb_ln_g=gm_c)
        mired_shift = cfg["mired_per_ln"] * rb_c
        tint = cfg["tint_per_ln"] * gm_c
        if a["raw"] and a.get("as_shot_kelvin"):
            new_mired = 1e6 / a["as_shot_kelvin"] + mired_shift
            s["Temperature"] = 1e6 / max(new_mired, 20)
            s["Tint"] = tint
            reasons.append(f"WB {a['as_shot_kelvin']:.0f}K -> {s['Temperature']:.0f}K ({a['wb_ref']})")
        else:
            s["IncrementalTemperature"] = -0.5 * mired_shift
            s["IncrementalTint"] = tint
            reasons.append(f"WB relative shift ({a['wb_ref']})")
    _add(s, cfg["fixed"])
    return s, meta
