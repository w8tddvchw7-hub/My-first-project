"""Step 1: analyse a single photo.

Reads the embedded preview from the RAW (fast - no demosaicing), EXIF data and
the camera's as-shot white balance, isolates the car and measures it.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .imageops import circular_mean_deg, linear_to_srgb, luminance, rgb_to_hsv, srgb_to_linear
from .mask import car_mask

# Proprietary RAW formats: Lightroom Classic reads edits from a NAME.xmp sidecar.
RAW_EXTS = {
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
    ".orf", ".rw2", ".rwl", ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".erf",
    ".mef", ".mos", ".kdc", ".dcr",
}
# Formats where Lightroom keeps XMP *inside* the file (sidecar is ignored).
EMBEDDED_EXTS = {".dng", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_EXTS = RAW_EXTS | EMBEDDED_EXTS


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS


# --------------------------------------------------------------------------- #
# Preview + as-shot white balance
# --------------------------------------------------------------------------- #

def _apply_libraw_flip(img: Image.Image, flip: int) -> Image.Image:
    if flip == 3:
        return img.transpose(Image.Transpose.ROTATE_180)
    if flip == 5:
        return img.transpose(Image.Transpose.ROTATE_90)
    if flip == 6:
        return img.transpose(Image.Transpose.ROTATE_270)
    return img


def xy_to_cct(x: float, y: float) -> float:
    """McCamy's approximation, chromaticity -> correlated colour temperature."""
    n = (x - 0.3320) / (0.1858 - y)
    return 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33


def _as_shot_kelvin(raw) -> float | None:
    """Estimate the camera's as-shot colour temperature from its WB multipliers."""
    try:
        wb = np.array(raw.camera_whitebalance[:3], dtype=float)
        cam_xyz = np.array(raw.rgb_xyz_matrix[:3], dtype=float)
        if wb.min() <= 0 or not np.any(cam_xyz):
            return None
        neutral = 1.0 / wb
        neutral /= neutral[1]
        xyz = np.linalg.solve(cam_xyz, neutral)
        s = xyz.sum()
        if s <= 0:
            return None
        k = xy_to_cct(xyz[0] / s, xyz[1] / s)
        return float(np.clip(k, 2000, 12000)) if math.isfinite(k) else None
    except Exception:
        return None


def load_preview(path: Path, max_side: int = 640) -> tuple[np.ndarray, float | None, str]:
    """Return (sRGB float image 0..1, as-shot Kelvin or None, preview source)."""
    ext = path.suffix.lower()
    kelvin = None
    if ext in RAW_EXTS or ext == ".dng":
        import rawpy

        with rawpy.imread(str(path)) as raw:
            kelvin = _as_shot_kelvin(raw)
            img = None
            source = "raw-embedded-jpeg"
            try:
                th = raw.extract_thumb()
                if th.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(th.data))
                    img.load()
                    if img.getexif().get(0x0112, 1) != 1:
                        img = ImageOps.exif_transpose(img)
                    else:
                        img = _apply_libraw_flip(img, raw.sizes.flip)
                else:
                    img = _apply_libraw_flip(Image.fromarray(th.data), raw.sizes.flip)
                if max(img.size) < 400:  # tiny thumbnail: not good enough
                    img = None
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                img = None
            if img is None:
                source = "raw-demosaic"
                arr = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                img = Image.fromarray(arr)
    else:
        source = "image"
        img = ImageOps.exif_transpose(Image.open(path))
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0, kelvin, source


# --------------------------------------------------------------------------- #
# EXIF
# --------------------------------------------------------------------------- #

def _parse_dt(s) -> datetime | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
        try:
            return datetime.strptime(s[:19] if "%z" not in fmt else s, fmt)
        except ValueError:
            continue
    return None


def _to_float(v) -> float | None:
    try:
        if hasattr(v, "num") and hasattr(v, "den"):
            return v.num / v.den if v.den else None
        if isinstance(v, str) and "/" in v:
            a, b = v.split("/", 1)
            return float(a) / float(b)
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_exif(path: Path) -> dict:
    out = {"iso": None, "exposure_time": None, "fnumber": None, "datetime": None,
           "make": None, "model": None}
    if shutil.which("exiftool"):
        try:
            res = subprocess.run(
                ["exiftool", "-j", "-n", "-ISO", "-ExposureTime", "-FNumber",
                 "-DateTimeOriginal", "-Make", "-Model", str(path)],
                capture_output=True, text=True, timeout=30, check=True,
            )
            d = json.loads(res.stdout)[0]
            out.update(iso=_to_float(d.get("ISO")), exposure_time=_to_float(d.get("ExposureTime")),
                       fnumber=_to_float(d.get("FNumber")), datetime=_parse_dt(d.get("DateTimeOriginal")),
                       make=d.get("Make"), model=d.get("Model"))
            return out
        except Exception:
            pass
    try:
        import exifread

        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        def first(name):
            t = tags.get(name)
            if t is None:
                return None
            vals = getattr(t, "values", None)
            return vals[0] if isinstance(vals, list) and vals else str(t)

        out.update(
            iso=_to_float(first("EXIF ISOSpeedRatings")),
            exposure_time=_to_float(first("EXIF ExposureTime")),
            fnumber=_to_float(first("EXIF FNumber")),
            datetime=_parse_dt(tags.get("EXIF DateTimeOriginal")),
            make=str(tags.get("Image Make", "")).strip() or None,
            model=str(tags.get("Image Model", "")).strip() or None,
        )
    except Exception:
        pass
    if out["iso"] is None or out["datetime"] is None:
        try:  # plain JPEG/TIFF
            with Image.open(path) as im:
                ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            out["iso"] = out["iso"] or _to_float(sub.get(0x8827))
            out["exposure_time"] = out["exposure_time"] or _to_float(sub.get(0x829A))
            out["fnumber"] = out["fnumber"] or _to_float(sub.get(0x829D))
            out["datetime"] = out["datetime"] or _parse_dt(sub.get(0x9003) or ex.get(0x0132))
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #

HUE_NAMES = [(18, "red"), (45, "orange"), (70, "yellow"), (165, "green"),
             (260, "blue"), (335, "purple"), (361, "red")]


def _hue_name(h: float) -> str:
    for limit, name in HUE_NAMES:
        if h < limit:
            return name
    return "red"


def _cast(lin: np.ndarray, sel: np.ndarray) -> tuple[float, float, int]:
    """ln(R/B) and ln(G/sqrt(RB)) of pixels that should be neutral grey."""
    n = int(sel.sum())
    if n < 150:
        return 0.0, 0.0, n
    r, g, b = (float(np.median(lin[..., c][sel])) + 1e-4 for c in range(3))
    return math.log(r / b), math.log(g / math.sqrt(r * b)), n


def measure(rgb: np.ndarray, mask: np.ndarray) -> dict:
    lin = srgb_to_linear(rgb)
    L = luminance(rgb)            # gamma-encoded luminance, how it "looks"
    Llin = luminance(lin)
    hue, sat, val = rgb_to_hsv(rgb)
    car = mask

    f = {
        "frame_mean": float(L.mean()),
        "frame_lin_med": float(np.median(Llin)),
        "frame_std": float(L.std()),
        "frame_p02": float(np.percentile(L, 2)),
        "frame_p98": float(np.percentile(L, 98)),
        "clip_hi": float((rgb.max(-1) > 0.985).mean()),
        "clip_lo": float((L < 0.02).mean()),
        "mask_frac": float(mask.mean()),
        # Paint brightness: the 70th percentile of the car, so the grille,
        # windows and tyres (dark, but not paint) do not drag it down.
        "car_L": float(np.percentile(L[car], 70)),
        "car_lin": float(np.percentile(Llin[car], 70)),
        "car_clip_hi": float((rgb.max(-1)[car] > 0.985).mean()),
    }

    # 1) Light colour: grey-world over the showroom (everything that is not
    #    the car), so a strong hall-light cast is measured in full.
    usable = (val > 0.10) & (val < 0.97)
    rb, gm, n = _cast(lin, ~car & usable & (sat < 0.6))
    ref = "background"
    if n < 150:
        rb, gm, n = _cast(lin, usable & (sat < 0.6))
        ref = "frame"

    # 2) Classify the car on a colour-corrected copy, so a white car under
    #    yellow light is still "white".
    gains = np.array([math.exp(-rb / 2 - gm / 3), math.exp(2 * gm / 3), math.exp(rb / 2 - gm / 3)])
    nrgb = linear_to_srgb(lin * gains)
    hue, sat, val = rgb_to_hsv(np.clip(nrgb, 0, 1))
    chroma = car & (sat > 0.30) & (val > 0.18)
    f["car_sat"] = float(np.median(sat[car]))
    f["car_chroma_frac"] = float(chroma.sum() / max(car.sum(), 1))
    f["car_hue"] = circular_mean_deg(hue[chroma], sat[chroma]) if chroma.sum() > 50 else None

    # Car tone, normalised for how exposed the frame is (a white car in an
    # under-exposed photo is still a white car).
    scene_white = max(f["frame_p98"], 0.30)
    f["car_L_norm"] = float(min(1.0, f["car_L"] / scene_white * 0.92))
    if f["car_chroma_frac"] > 0.35 and f["car_hue"] is not None:
        name = _hue_name(f["car_hue"])
        tone = "red" if name == "red" else "color"
    else:
        v = f["car_L_norm"]
        name = tone = "black" if v < 0.25 else "dark" if v < 0.40 else "silver" if v < 0.64 else "white"
    f["car_color"] = name
    f["car_tone"] = tone

    # 3) White / silver paint is the best neutral reference there is: refine
    #    the white balance on the paint itself so white never turns yellow.
    if tone in ("white", "silver"):
        paint = car & (sat < 0.20) & (val > 0.20) & (val < 0.97)
        prb, pgm, pn = _cast(lin, paint)
        if pn >= 150:
            rb, gm, n, ref = prb, pgm, pn, "car-paint"
    f.update(cast_rb=rb, cast_g=gm, wb_ref=ref, wb_ref_pixels=n)
    return f


def analyze(path: Path, mask_method: str = "auto", keep_preview: bool = False) -> dict:
    path = Path(path)
    rgb, kelvin, source = load_preview(path)
    # Normalise brightness before isolating the car, so dark frames mask well.
    p98 = float(np.percentile(rgb.max(-1), 98))
    mask, mask_src = car_mask(np.clip(rgb * (0.92 / max(p98, 0.05)), 0, 1), mask_method)
    a = {"file": str(path), "raw": is_raw(path), "preview_source": source,
         "as_shot_kelvin": kelvin, "mask_source": mask_src}
    a.update(read_exif(path))
    a.update(measure(rgb, mask))

    # Interior / detail shot (seats, dashboard, wheel): the frame is mostly
    # dark cabin, unlike a bright showroom exterior.
    # Compared with the frame's own highlights, so an under-exposed exterior
    # (dark everywhere) is not mistaken for a cabin (dark next to bright windows).
    p98_lin = float(srgb_to_linear(np.array(a["frame_p98"])))
    a["scene"] = "interior" if a["frame_lin_med"] < 0.12 * max(p98_lin, 0.05) else "exterior"
    hour = a["datetime"].hour if a["datetime"] else None
    iso = a["iso"] or 100
    a["evening"] = bool(hour is not None and (hour >= 18 or hour < 6))
    a["artificial_light"] = bool((kelvin is not None and kelvin < 4300) or a["cast_rb"] > 0.12)
    a["low_light"] = bool(iso >= 1600 or a["frame_mean"] < 0.22)
    if keep_preview:
        a["_rgb"] = rgb
        a["_mask"] = mask
    return a


def jsonable(a: dict) -> dict:
    out = {}
    for k, v in a.items():
        if k.startswith("_"):
            continue
        out[k] = v.isoformat() if isinstance(v, datetime) else v
    return out
