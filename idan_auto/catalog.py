"""Read past edits out of a Lightroom Classic catalog (.lrcat is SQLite).

Works on a temporary *copy* of the catalog, opened read-only, so it is safe
even while Lightroom is running.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

_PAIR = re.compile(
    r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false)'
)


def parse_lua_settings(text: str) -> dict:
    """Parse the top level of Lightroom's `s = { Exposure2012 = 0.45, ... }`."""
    if not text:
        return {}
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return {}
    body = text[a + 1:b]
    prev = None
    while prev != body:  # drop nested tables (tone curves, masks...)
        prev = body
        body = re.sub(r"\{[^{}]*\}", "", body)
    out = {}
    for k, v in _PAIR.findall(body):
        if v.startswith('"'):
            out[k] = v[1:-1]
        elif v in ("true", "false"):
            out[k] = v == "true"
        else:
            out[k] = float(v)
    return out


QUERY = """
SELECT i.id_local, rf.absolutePath, fo.pathFromRoot, f.baseName, f.extension,
       ds.text, i.captureTime
FROM Adobe_images i
JOIN AgLibraryFile f        ON i.rootFile = f.id_local
JOIN AgLibraryFolder fo     ON f.folder = fo.id_local
JOIN AgLibraryRootFolder rf ON fo.rootFolder = rf.id_local
JOIN Adobe_imageDevelopSettings ds ON ds.image = i.id_local
{where}
ORDER BY i.captureTime DESC
"""


def iter_catalog_edits(lrcat: Path, path_map: list[tuple[str, str]] | None = None):
    lrcat = Path(lrcat)
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "catalog.lrcat"
        shutil.copy2(lrcat, dst)
        for suffix in ("-wal", "-shm"):
            side = lrcat.with_name(lrcat.name + suffix)
            if side.exists():
                shutil.copy2(side, dst.with_name(dst.name + suffix))
        con = sqlite3.connect(str(dst))
        try:
            try:  # skip virtual copies when the column exists
                rows = con.execute(QUERY.format(where="WHERE i.masterImage IS NULL")).fetchall()
            except sqlite3.OperationalError:
                rows = con.execute(QUERY.format(where="")).fetchall()
        finally:
            con.close()
    for _id, root, sub, base, ext, text, capture in rows:
        path = f"{root or ''}{sub or ''}{base}.{ext}"
        for old, new in path_map or []:
            if path.startswith(old):
                path = new + path[len(old):]
                break
        yield {"id": _id, "path": path, "capture_time": capture,
               "settings": parse_lua_settings(text or "")}
