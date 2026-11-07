"""Idan Auto: per-photo Lightroom Classic edits for dealership car photos.

The tool never touches the RAW files. It analyses each photo (EXIF, histogram,
light colour and - most importantly - the car body itself) and writes an XMP
sidecar next to it. Lightroom Classic reads that sidecar on import (or via
Metadata > Read Metadata from Files) and the photo opens already edited.
"""

__version__ = "0.1.0"
