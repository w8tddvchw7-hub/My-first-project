"""Rough before/after preview of the computed settings, for the HTML report.

This only approximates Lightroom's rendering - it is for checking the
direction and strength of each decision quickly, not for final judgement.
"""

from __future__ import annotations

import math

import numpy as np

from .imageops import linear_to_srgb, luminance, rgb_to_hsv, srgb_to_linear


def simulate(rgb: np.ndarray, s: dict, meta: dict) -> np.ndarray:
    lin = srgb_to_linear(rgb).astype(np.float64)
    lin *= 2.0 ** s.get("Exposure2012", 0.0)
    rb, g = meta.get("wb_ln_rb", 0.0), meta.get("wb_ln_g", 0.0)
    lin[..., 0] *= math.exp(-rb / 2)
    lin[..., 2] *= math.exp(rb / 2)
    lin[..., 1] *= math.exp(-g)
    v = linear_to_srgb(lin)

    L = np.maximum(luminance(v), 1e-4)
    hl = s.get("Highlights2012", 0) / 100
    sh = s.get("Shadows2012", 0) / 100
    wh = s.get("Whites2012", 0) / 100
    bl = s.get("Blacks2012", 0) / 100
    ct = s.get("Contrast2012", 0) / 100 + s.get("Dehaze", 0) / 250
    t = np.clip((L - 0.45) / 0.8, 0, 1)
    L2 = L + hl * 0.35 * t**1.5 * np.maximum(L - 0.45, 0) * 2
    L2 = L2 + sh * 0.25 * np.clip(1 - L2 / 0.55, 0, 1) ** 2 * np.minimum(L2 * 4, 1)
    L2 = L2 + wh * 0.10 * L2**3 + bl * 0.08 * (1 - np.clip(L2, 0, 1)) ** 4
    L2 = 0.5 + (L2 - 0.5) * (1 + ct * 0.6)
    L2 = np.where(L2 > 0.9, 0.9 + 0.1 * np.tanh((L2 - 0.9) / 0.1), L2)
    v = v * (np.clip(L2, 0, 1) / L)[..., None]

    hue, sat, _ = rgb_to_hsv(np.clip(v, 0, 1))
    gray = luminance(v)[..., None]
    factor = 1 + s.get("Saturation", 0) / 100 + s.get("Vibrance", 0) / 100 * (1 - sat) * 0.8
    red = ((hue < 20) | (hue > 340)).astype(float)
    yel = ((hue > 35) & (hue < 75)).astype(float)
    factor = factor + red * s.get("SaturationAdjustmentRed", 0) / 100 \
        + yel * s.get("SaturationAdjustmentYellow", 0) / 100
    v = gray + (v - gray) * np.clip(factor, 0, 2)[..., None]
    v = v * (1 + red * s.get("LuminanceAdjustmentRed", 0) / 300)[..., None]
    return np.clip(v, 0, 1)
