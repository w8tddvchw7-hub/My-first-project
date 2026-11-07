"""Small numpy colour helpers shared by the analyser and the simulator."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, None)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB (0..1) -> (hue degrees, saturation, value)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    safe = np.maximum(d, 1e-6)
    h = np.where(
        mx == r,
        ((g - b) / safe) % 6,
        np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4),
    )
    h = np.where(d > 1e-6, h * 60.0, 0.0)
    return h, s, mx


def circular_mean_deg(h: np.ndarray, weights: np.ndarray | None = None) -> float:
    rad = np.deg2rad(h)
    w = np.ones_like(rad) if weights is None else weights
    ang = np.arctan2((np.sin(rad) * w).sum(), (np.cos(rad) * w).sum())
    return float(np.rad2deg(ang) % 360)
