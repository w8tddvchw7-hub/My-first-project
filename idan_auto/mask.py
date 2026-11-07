"""Isolate the car so we measure the paint, not the showroom.

Uses `rembg` (AI background removal) when it is installed. Otherwise falls back
to a heuristic: anything in the central part of the frame whose colour differs
from the colours found along the frame edges (walls, floor, ceiling).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

_REMBG_SESSION = None


def _rembg_mask(rgb: np.ndarray) -> np.ndarray:
    global _REMBG_SESSION
    from rembg import new_session, remove  # optional dependency

    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("u2net")
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    out = remove(img, session=_REMBG_SESSION, only_mask=True)
    return np.asarray(out.convert("L")) > 127


def _kmeans(points: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(points))
    centers = points[rng.choice(len(points), k, replace=False)]
    for _ in range(iters):
        d = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for i in range(k):
            sel = points[lab == i]
            if len(sel):
                centers[i] = sel.mean(0)
    return centers


def _smooth(mask: np.ndarray, size: int = 5) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))  # close
    img = img.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))  # open
    img = img.filter(ImageFilter.MedianFilter(size))
    return np.asarray(img) > 127


def center_prior(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy / max(h - 1, 1)
    xx = xx / max(w - 1, 1)
    return ((xx - 0.5) / 0.46) ** 2 + ((yy - 0.58) / 0.40) ** 2 <= 1.0


def _heuristic_mask(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    h, w = rgb.shape[:2]
    prior = center_prior(h, w)
    bw = max(2, int(min(h, w) * 0.05))
    border = np.concatenate(
        [
            rgb[:bw].reshape(-1, 3),
            rgb[-bw:].reshape(-1, 3),
            rgb[:, :bw].reshape(-1, 3),
            rgb[:, -bw:].reshape(-1, 3),
        ]
    )
    step = max(1, len(border) // 4000)
    border = border[::step].astype(np.float64)
    centers = _kmeans(border.copy(), k=8)

    flat = rgb.reshape(-1, 3).astype(np.float64)
    dist = np.sqrt(((flat[:, None, :] - centers[None]) ** 2).sum(-1)).min(1).reshape(h, w)
    bdist = np.sqrt(((border[:, None, :] - centers[None]) ** 2).sum(-1)).min(1)
    thr = max(0.07, float(np.percentile(bdist, 95)) * 1.5)

    mask = _smooth((dist > thr) & prior)
    if mask.mean() < 0.10:  # a dealership shot: the car fills the frame
        return prior, "center-prior"
    return mask, "heuristic"


def car_mask(rgb: np.ndarray, method: str = "auto") -> tuple[np.ndarray, str]:
    """Return (boolean mask of the car, which method produced it)."""
    if method in ("auto", "rembg"):
        try:
            m = _rembg_mask(rgb)
            if m.mean() > 0.02:
                return m, "rembg"
        except ImportError:
            if method == "rembg":
                raise SystemExit("rembg is not installed: pip install rembg")
        except Exception:  # model download failure etc. -> fall back
            if method == "rembg":
                raise
    return _heuristic_mask(rgb)
