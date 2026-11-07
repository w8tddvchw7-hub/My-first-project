"""Render finished JPEGs directly from the computed settings (no Lightroom).

For when there is no Lightroom Classic at hand (e.g. working from a phone
with JPEGs). It applies the same decisions as the XMP - white balance,
exposure, highlights/shadows, clarity, HSL - with a local tone mapper so
blown ceilings and windows come back without flattening the car.
Lightroom on the RAW will always have more latitude than this.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from .imageops import linear_to_srgb, luminance, rgb_to_hsv, srgb_to_linear

MID = math.log2(0.18)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Fast large-radius Gaussian: downscale, box-blur x3, upscale."""
    h, w = a.shape
    f = max(1, int(sigma / 6))
    small = Image.fromarray(a.astype(np.float32), mode="F").resize(
        (max(1, w // f), max(1, h // f)), Image.BILINEAR)
    s = np.asarray(small, dtype=np.float64)
    r = max(1, int(round(sigma / f)))
    for _ in range(3):
        for axis in (0, 1):
            c = np.cumsum(np.pad(s, [(r + 1, r) if i == axis else (0, 0) for i in range(2)],
                                 mode="edge"), axis=axis)
            s = (np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis)
                 - np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)) / (2 * r + 1)
    return np.asarray(Image.fromarray(s.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR))


def render(rgb: np.ndarray, s: dict, meta: dict) -> np.ndarray:
    """rgb: float sRGB 0..1 (full resolution). Returns float sRGB 0..1."""
    lin = srgb_to_linear(rgb).astype(np.float64)

    # White balance + exposure.
    rb, g = meta.get("wb_ln_rb", 0.0), meta.get("wb_ln_g", 0.0)
    gains = np.array([math.exp(-rb / 2 - g / 3), math.exp(2 * g / 3), math.exp(rb / 2 - g / 3)])
    lin *= gains * 2.0 ** s.get("Exposure2012", 0.0)

    # Local tone mapping in log space: base (large scale) + detail.
    Y = np.maximum(luminance(lin), 1e-5)
    logY = np.log2(Y)
    sigma = 0.012 * math.hypot(*Y.shape)
    base = _blur(logY, sigma)
    # Edge-aware: strong edges (dark wheel against a bright window) belong to
    # the base layer, otherwise they turn into glowing halos.
    detail = np.clip(logY - base, -0.7, 0.7)
    base = logY - detail

    b = base - MID                        # stops relative to mid grey
    hl = s.get("Highlights2012", 0) / 100
    sh = s.get("Shadows2012", 0) / 100
    pos = np.maximum(b, 0)
    neg = np.maximum(-b, 0)
    b = b + hl * 0.45 * pos * np.clip(pos / 2.5, 0, 1)
    b = b + sh * 0.40 * neg * np.clip(neg / 3.5, 0, 1)
    clarity = s.get("Clarity2012", 0) / 100 * 0.6 + s.get("Texture", 0) / 100 * 0.25 \
        + s.get("Dehaze", 0) / 100 * 0.3
    newlog = b + MID + detail * (1 + clarity)
    lin *= (2.0 ** newlog / Y)[..., None]

    # Highlight shoulder (Whites) - extended Reinhard, midtones compensated.
    white = 2.0 ** (1.3 + s.get("Whites2012", 0) / 100 * 0.8)
    comp = 1.12
    x = lin * comp
    lin = x * (1 + x / white**2) / (1 + x)
    v = linear_to_srgb(np.clip(lin, 0, None))
    v = np.clip(v, 0, 1)

    # Contrast (S-curve around mid) and Blacks.
    c = s.get("Contrast2012", 0) / 100 * 1.2 + s.get("Dehaze", 0) / 100 * 0.6
    sm = v * v * (3 - 2 * v)
    v = v + c * (sm - v)
    v = v + s.get("Blacks2012", 0) / 100 * 0.12 * (1 - v) ** 4
    v = np.clip(v, 0, 1)

    # Colour: vibrance / saturation / HSL red & yellow & orange.
    hue, sat, _ = rgb_to_hsv(v)
    gray = luminance(v)[..., None]
    fac = 1 + s.get("Saturation", 0) / 100 + s.get("Vibrance", 0) / 100 * (1 - sat) ** 2 * 0.9
    red = np.clip(1 - np.minimum(np.abs(hue - 0), np.abs(hue - 360)) / 20, 0, 1)
    ora = np.clip(1 - np.abs(hue - 30) / 18, 0, 1)
    yel = np.clip(1 - np.abs(hue - 55) / 20, 0, 1)
    fac = fac + red * s.get("SaturationAdjustmentRed", 0) / 100 \
        + ora * s.get("SaturationAdjustmentOrange", 0) / 100 \
        + yel * s.get("SaturationAdjustmentYellow", 0) / 100
    v = gray + (v - gray) * np.clip(fac, 0, 2)[..., None]
    v = v * (1 + red * s.get("LuminanceAdjustmentRed", 0) / 250)[..., None]

    # Near-white areas of the source (windows, ceiling lights) carry little
    # detail in a JPEG; darkening them only makes grey patches. The brighter a
    # source pixel was, the less the edit may darken it - a smooth ramp, so no
    # visible boundaries.
    src = rgb.astype(np.float64)
    w = np.clip((_blur(luminance(src), 1.5) - 0.72) / 0.26, 0, 1)
    w = (w * w * (3 - 2 * w))[..., None]
    v = v * (1 - w) + np.maximum(v, src) * w
    return np.clip(v, 0, 1)


def render_file(src, dst, s: dict, meta: dict, quality: int = 95) -> None:
    from PIL import ImageOps

    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    out = render(np.asarray(img, dtype=np.float32) / 255.0, s, meta)
    im = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
    amount = int(s.get("Sharpness", 40) * 2.2)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=amount, threshold=2))
    im.save(dst, quality=quality, subsampling=0)
