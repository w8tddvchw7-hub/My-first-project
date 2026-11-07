"""Command line.

  python -m idan_auto run   PHOTOS_DIR [--model idan_style.json] [--report]
  python -m idan_auto learn CATALOG.lrcat --out idan_style.json
  python -m idan_auto analyze FILE...
  python -m idan_auto config > my_config.json
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .analyze import EMBEDDED_EXTS, SUPPORTED_EXTS, analyze, jsonable
from .model import StyleModel, blend, feature_vector, targets_from_settings
from .rules import DEFAULT_CONFIG, clamp_and_round, compute_settings, load_config
from .xmp import embed_with_exiftool, write_sidecar

KEY_COLS = ["Exposure2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
            "Temperature", "Tint", "SaturationAdjustmentRed", "LuminanceSmoothing"]


def find_photos(folder: Path, recursive: bool) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                  and not p.name.startswith("."))


def process(path: Path, cfg: dict, model: StyleModel | None, args, keep_preview=False):
    a = analyze(path, args.mask, keep_preview=keep_preview)
    raw_s, meta = compute_settings(a, cfg)
    if model is not None and len(model):
        raw_s, mmeta = blend(raw_s, a, model, args.model_weight)
        meta.update(mmeta)
    return a, clamp_and_round(raw_s), meta


def cmd_run(args) -> int:
    folder = Path(args.folder)
    cfg = load_config(args.config)
    if args.target_ev:
        cfg["exposure_bias"] += args.target_ev
    model = StyleModel.load(Path(args.model)) if args.model else None
    photos = find_photos(folder, args.recursive)
    if args.include_jpeg is False:
        photos = [p for p in photos if p.suffix.lower() not in {".jpg", ".jpeg"}]
    if not photos:
        print(f"No photos found in {folder}", file=sys.stderr)
        return 1

    report_dir = Path(args.report_dir or Path.cwd() / "idan_auto_reports" / folder.name) if args.report else None
    if report_dir:
        (report_dir / "img").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(photos, 1):
        try:
            a, s, meta = process(p, cfg, model, args, keep_preview=bool(report_dir))
        except Exception as e:  # keep going on a bad file
            print(f"[{i}/{len(photos)}] {p.name}: ERROR {e}", file=sys.stderr)
            continue
        if args.dry_run:
            status = "dry-run"
        elif a["raw"]:
            _, status = write_sidecar(p, s, force=args.force)
        elif p.suffix.lower() in EMBEDDED_EXTS and args.embed:
            status = embed_with_exiftool(p, s)
        else:
            status = "analysed only (Lightroom ignores sidecars for DNG/JPEG; see --embed)"
        why = "; ".join(meta["reasons"])
        if "model_weight" in meta:
            why += f"; style model weight {meta['model_weight']}"
        print(f"[{i}/{len(photos)}] {p.name}: {a['car_color']}/{a['car_tone']} "
              f"EV {s['Exposure2012']:+.2f} -> {status}\n      {why}")
        rows.append({"analysis": jsonable(a), "settings": s, "meta": meta, "status": status})
        if report_dir:
            _save_report_images(report_dir, i, p, a, s, meta)

    if report_dir:
        (report_dir / "analysis.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        _write_html(report_dir, rows)
        print(f"\nReport: {report_dir / 'report.html'}")
    return 0


def _save_report_images(d: Path, i: int, p: Path, a: dict, s: dict, meta: dict) -> None:
    from .simulate import simulate

    rgb, mask = a.pop("_rgb"), a.pop("_mask")
    after = simulate(rgb, s, meta)
    overlay = rgb.copy()
    overlay[mask] = overlay[mask] * 0.55 + np.array([0.0, 0.8, 0.3]) * 0.45
    strip = np.concatenate([rgb, after, overlay], axis=1)
    Image.fromarray((strip * 255).astype(np.uint8)).save(d / "img" / f"{i:04d}_{p.stem}.jpg", quality=85)


def _write_html(d: Path, rows: list[dict]) -> None:
    trs = []
    for i, r in enumerate(rows, 1):
        a, s = r["analysis"], r["settings"]
        stem = Path(a["file"]).stem
        vals = "".join(f"<td>{s.get(k, '')}</td>" for k in KEY_COLS)
        trs.append(
            f"<tr><td class=name>{html.escape(Path(a['file']).name)}<br><small>"
            f"{html.escape(a['car_color'])} / {html.escape(a['car_tone'])}"
            f"{' · evening' if a['evening'] else ''}{' · hall light' if a['artificial_light'] else ''}"
            f"<br>ISO {a.get('iso') or '?'} · mask: {a['mask_source']}</small></td>"
            f"<td><img src='img/{i:04d}_{html.escape(stem)}.jpg'></td>{vals}"
            f"<td class=why>{html.escape('; '.join(r['meta']['reasons']))}</td></tr>")
    head = "".join(f"<th>{k.replace('2012', '')}</th>" for k in KEY_COLS)
    (d / "report.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Idan Auto report</title>
<style>body{{font:13px system-ui;margin:16px;background:#111;color:#eee}}
table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #333;padding:6px;vertical-align:top}}
img{{width:720px;max-width:60vw}}.why{{max-width:260px;color:#aaa}}small{{color:#aaa}}</style>
<h1>Idan Auto</h1><p>Each image: original preview · rough simulation of the edit · detected car (green).
The simulation is approximate - judge final results in Lightroom.</p>
<table><tr><th>File</th><th>Before · After · Mask</th>{head}<th>Why</th></tr>{''.join(trs)}</table>""",
                                encoding="utf-8")


def cmd_learn(args) -> int:
    from .catalog import iter_catalog_edits

    path_map = [tuple(m.split("=", 1)) for m in args.path_map or []]
    X, Y, files = [], [], []
    seen = missing = unedited = errors = 0
    for e in iter_catalog_edits(Path(args.catalog), path_map):
        p = Path(e["path"])
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        seen += 1
        if not p.exists():
            missing += 1
            continue
        try:
            a = analyze(p, args.mask)
        except Exception as ex:
            errors += 1
            print(f"  {p.name}: {ex}", file=sys.stderr)
            continue
        t = targets_from_settings(e["settings"], a)
        if t is None:
            unedited += 1
            continue
        X.append(feature_vector(a))
        Y.append(t)
        files.append(str(p))
        print(f"  [{len(X)}] {p.name}: {a['car_tone']}")
        if args.limit and len(X) >= args.limit:
            break
    print(f"catalog photos: {seen}, learned: {len(X)}, missing on disk: {missing}, "
          f"unedited: {unedited}, errors: {errors}")
    if not X:
        print("Nothing to learn from. Check paths (--path-map OLD=NEW).", file=sys.stderr)
        return 1
    StyleModel(X, Y, files).save(Path(args.out))
    print(f"Saved style model -> {args.out}")
    return 0


def cmd_analyze(args) -> int:
    cfg = load_config(args.config)
    for f in args.files:
        a = analyze(Path(f), args.mask)
        s, meta = compute_settings(a, cfg)
        print(json.dumps({"analysis": jsonable(a), "settings": clamp_and_round(s),
                          "reasons": meta["reasons"]}, indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="idan_auto", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mask_help = "car isolation: auto (rembg if installed), rembg, heuristic"

    r = sub.add_parser("run", help="analyse a folder and write XMP sidecars")
    r.add_argument("folder")
    r.add_argument("--recursive", "-r", action="store_true")
    r.add_argument("--model", help="style model from `learn`")
    r.add_argument("--model-weight", type=float, help="force model weight 0..1")
    r.add_argument("--config", help="JSON overrides for the rules")
    r.add_argument("--target-ev", type=float, default=0.0, help="shift all exposures (EV)")
    r.add_argument("--force", action="store_true", help="overwrite existing .xmp (backup kept)")
    r.add_argument("--dry-run", action="store_true", help="analyse only, write nothing")
    r.add_argument("--embed", action="store_true", help="embed into DNG/JPEG/TIFF via exiftool")
    r.add_argument("--include-jpeg", action="store_true", default=False,
                   help="also process JPEG files (analysis/embedding)")
    r.add_argument("--report", action="store_true", help="HTML report with before/after")
    r.add_argument("--report-dir")
    r.add_argument("--mask", default="auto", help=mask_help)
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("learn", help="learn your style from a Lightroom catalog")
    l.add_argument("catalog")
    l.add_argument("--out", default="idan_style.json")
    l.add_argument("--path-map", action="append", metavar="OLD=NEW",
                   help="rewrite catalog path prefixes (e.g. moved drive)")
    l.add_argument("--limit", type=int, default=0)
    l.add_argument("--mask", default="auto", help=mask_help)
    l.set_defaults(func=cmd_learn)

    z = sub.add_parser("analyze", help="print the analysis and settings for files")
    z.add_argument("files", nargs="+")
    z.add_argument("--config")
    z.add_argument("--mask", default="auto", help=mask_help)
    z.set_defaults(func=cmd_analyze)

    c = sub.add_parser("config", help="print the default rule config as JSON")
    c.set_defaults(func=lambda a: print(json.dumps(DEFAULT_CONFIG, indent=2)) or 0)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
