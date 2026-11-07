"""Learn your style": a nearest-neighbour model over your past edits.

For each photo you already edited we store its analysis features and the
slider values you chose. For a new photo we find the most similar past photos
(similar car tone, brightness, light colour, ISO...) and average what you did
there. The result is blended with the rules - the more examples, the more the
model wins.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FEATURES = [
    "log_car_lin", "car_L_norm", "car_sat", "car_chroma_frac", "hue_sin", "hue_cos",
    "frame_mean", "frame_std", "clip_hi", "clip_lo", "cast_rb", "cast_g",
    "log_iso", "evening", "artificial_light",
]

TARGETS = [
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012",
    "Blacks2012", "Texture", "Clarity2012", "Dehaze", "Vibrance", "Saturation",
    "SaturationAdjustmentRed", "LuminanceAdjustmentRed", "SaturationAdjustmentOrange",
    "SaturationAdjustmentYellow", "LuminanceSmoothing", "TempMiredDelta", "Tint",
]

CORE = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012",
        "Blacks2012", "Clarity2012", "Vibrance", "Texture", "Dehaze"]


def feature_vector(a: dict) -> list[float]:
    h = a.get("car_hue")
    chroma = a.get("car_chroma_frac", 0.0)
    hs = math.sin(math.radians(h)) * chroma if h is not None else 0.0
    hc = math.cos(math.radians(h)) * chroma if h is not None else 0.0
    return [
        math.log2(max(a["car_lin"], 1e-4)), a["car_L_norm"], a["car_sat"], chroma, hs, hc,
        a["frame_mean"], a["frame_std"], a["clip_hi"], a["clip_lo"], a["cast_rb"], a["cast_g"],
        math.log2(max(a.get("iso") or 100, 50) / 100), float(a["evening"]),
        float(a["artificial_light"]),
    ]


def targets_from_settings(st: dict, a: dict) -> list[float] | None:
    """Slider values from a catalog entry; None if the photo looks unedited."""
    if sum(1 for k in CORE if abs(float(st.get(k, 0) or 0)) > 1e-6) < 2:
        return None
    row = []
    for k in TARGETS:
        if k == "TempMiredDelta":
            wb = st.get("WhiteBalance", "As Shot")
            if wb == "As Shot":
                row.append(0.0)
            elif st.get("Temperature") and a.get("as_shot_kelvin"):
                row.append(1e6 / float(st["Temperature"]) - 1e6 / a["as_shot_kelvin"])
            else:
                row.append(float("nan"))
        elif k == "Tint":
            row.append(float(st["Tint"]) if st.get("WhiteBalance", "As Shot") != "As Shot"
                       and "Tint" in st else float("nan"))
        else:
            v = st.get(k)
            row.append(float(v) if isinstance(v, (int, float)) else 0.0)
    return row


class StyleModel:
    def __init__(self, X, Y, files=None, k: int = 7):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.files = files or []
        self.k = k
        self.mean = self.X.mean(0)
        self.std = self.X.std(0) + 1e-6

    def __len__(self):
        return len(self.X)

    def predict(self, a: dict) -> tuple[dict, float]:
        """Return (predicted sliders, mean normalised distance to neighbours)."""
        x = (np.asarray(feature_vector(a)) - self.mean) / self.std
        Xn = (self.X - self.mean) / self.std
        d = np.sqrt(((Xn - x) ** 2).sum(1))
        idx = np.argsort(d)[: min(self.k, len(d))]
        w = 1.0 / (d[idx] + 0.25)
        pred = {}
        for j, key in enumerate(TARGETS):
            col = self.Y[idx, j]
            ok = ~np.isnan(col)
            if ok.any():
                pred[key] = float((col[ok] * w[ok]).sum() / w[ok].sum())
        return pred, float(d[idx].mean())

    def weight(self, dist: float, override: float | None = None) -> float:
        if override is not None:
            return override
        n = len(self)
        return min(0.85, n / (n + 30)) * math.exp(-dist / 4.0)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "version": 1, "features": FEATURES, "targets": TARGETS, "k": self.k,
            "X": self.X.tolist(),
            "Y": [[None if math.isnan(v) else v for v in r] for r in self.Y.tolist()],
            "files": self.files,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StyleModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("features") != FEATURES or d.get("targets") != TARGETS:
            raise SystemExit("Model was built by a different version - run `learn` again.")
        Y = [[float("nan") if v is None else v for v in r] for r in d["Y"]]
        return cls(d["X"], Y, d.get("files"), d.get("k", 7))


def blend(rule_settings: dict, a: dict, model: StyleModel, override: float | None = None):
    pred, dist = model.predict(a)
    w = model.weight(dist, override)
    out = dict(rule_settings)
    for k, v in pred.items():
        if k == "TempMiredDelta":
            if a.get("raw") and a.get("as_shot_kelvin"):
                base = 1e6 / out.get("Temperature", a["as_shot_kelvin"])
                learned = 1e6 / a["as_shot_kelvin"] + v
                out["Temperature"] = 1e6 / max(20, base * (1 - w) + learned * w)
            continue
        if k == "Tint":
            if "Temperature" in out:
                out["Tint"] = out.get("Tint", 0) * (1 - w) + v * w
            continue
        out[k] = out.get(k, 0) * (1 - w) + v * w
    return out, {"model_weight": round(w, 2), "model_distance": round(dist, 2)}
