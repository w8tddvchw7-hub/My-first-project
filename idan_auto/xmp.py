"""Step 3: write the XMP sidecar Lightroom Classic reads."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import quoteattr

from . import __version__

TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Idan Auto {version}">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   xmp:CreatorTool="Idan Auto {version}"
   crs:Version="15.0"
   crs:ProcessVersion="11.0"
{attrs}
   crs:HasSettings="True"
   crs:AlreadyApplied="False"/>
 </rdf:RDF>
</x:xmpmeta>
"""


def _fmt(k: str, v) -> str:
    if k == "Exposure2012":
        return f"{v:+.2f}"
    if isinstance(v, float):
        return f"{v:.1f}"
    if isinstance(v, int) and k not in ("Temperature", "LensProfileEnable", "AutoLateralCA") and v > 0:
        return f"+{v}"
    return str(v)


def build_xmp(settings: dict) -> str:
    s = dict(settings)
    lines = []
    if "Temperature" in s:
        lines.append('   crs:WhiteBalance="Custom"')
    elif "IncrementalTemperature" in s:
        lines.append('   crs:WhiteBalance="Custom"')
    if s.get("LensProfileEnable"):
        lines.append('   crs:LensProfileSetup="LensDefaults"')
    for k in sorted(s):
        lines.append(f"   crs:{k}={quoteattr(_fmt(k, s[k]))}")
    return TEMPLATE.format(version=__version__, attrs="\n".join(lines))


def sidecar_path(photo: Path) -> Path:
    return photo.with_suffix(".xmp")


def write_sidecar(photo: Path, settings: dict, force: bool = False) -> tuple[Path, str]:
    """Write NAME.xmp next to the photo. Returns (path, status)."""
    xmp = sidecar_path(photo)
    status = "written"
    if xmp.exists():
        if not force:
            return xmp, "skipped (xmp exists, use --force)"
        bak = xmp.with_suffix(".xmp.bak")
        shutil.copy2(xmp, bak)
        status = f"overwritten (backup: {bak.name})"
    xmp.write_text(build_xmp(settings), encoding="utf-8")
    return xmp, status


def embed_with_exiftool(photo: Path, settings: dict) -> str:
    """DNG/JPEG/TIFF keep XMP inside the file; needs exiftool. Modifies the file."""
    if not shutil.which("exiftool"):
        return "skipped (DNG/JPEG need exiftool for --embed)"
    tmp = photo.with_name(photo.name + ".idan.xmp")
    tmp.write_text(build_xmp(settings), encoding="utf-8")
    try:
        subprocess.run(["exiftool", "-q", "-overwrite_original_in_place",
                        "-tagsfromfile", str(tmp), "-xmp-crs:all", str(photo)],
                       check=True, capture_output=True, timeout=60)
        return "embedded"
    finally:
        tmp.unlink(missing_ok=True)
