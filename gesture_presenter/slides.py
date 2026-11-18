"""Slide loading: runs once at start-up, before the per-frame pipeline.

Turns any supported presentation file into a list of slide images (numpy BGR
arrays) that OpenCV can draw. PDF files are rendered directly with PyMuPDF.
PowerPoint and OpenDocument decks are first converted to PDF with LibreOffice;
Keynote files are exported by Keynote itself on macOS. Each conversion is
cached by the file's content hash, so the second open is instant.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# File extensions we can open.
SUPPORTED = (".pdf", ".pptx", ".ppt", ".ppsx", ".pps", ".odp", ".key")
# Folder where converted PDFs are kept between runs.
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "gesture-presenter")
# Usual LibreOffice install locations, checked if `soffice` is not on the PATH.
SOFFICE_PATHS = ("/Applications/LibreOffice.app/Contents/MacOS/soffice", "/opt/homebrew/bin/soffice",
                 r"C:\Program Files\LibreOffice\program\soffice.exe")

HASH_LENGTH = 16                # hex characters of the SHA-1 used as the cache file name
KEYNOTE_TIMEOUT_S = 180         # give up on a Keynote export after this many seconds
LIBREOFFICE_TIMEOUT_S = 300     # give up on a LibreOffice conversion after this many seconds
RENDER_ZOOM = 2.0               # PDF render scale (2x = sharp slides on a 1080p screen)
DIALOG_TITLE = "Choose a presentation"


def _soffice() -> str | None:
    """Return the path to LibreOffice's command-line program, or None."""
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for path in SOFFICE_PATHS:
        if os.path.exists(path):
            return path
    return None


def _keynote_to_pdf(src: str, out: str) -> None:
    """Ask the Keynote app (macOS only) to export `src` as a PDF at `out`."""
    script = (f'tell application "Keynote"\nset d to open POSIX file "{src}"\n'
              f'export d to POSIX file "{out}" as PDF\nclose d saving no\nend tell')
    subprocess.run(["osascript", "-e", script], check=True, capture_output=True,
                   timeout=KEYNOTE_TIMEOUT_S)


def _file_hash(path: str) -> str:
    """Short SHA-1 of the file's bytes, used as its cache key."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:HASH_LENGTH]


def _libreoffice_to_pdf(path: str, out: str) -> None:
    """Convert an office document to a PDF at `out` using LibreOffice."""
    soffice = _soffice()
    if not soffice:
        raise RuntimeError("LibreOffice is needed to open PowerPoint files.\n"
                           "  macOS:  brew install --cask libreoffice\n"
                           "  Or export your deck as PDF and open that instead.")
    with tempfile.TemporaryDirectory() as tmp:
        # A separate profile folder lets the conversion run even if the user
        # already has LibreOffice open.
        command = [soffice, f"-env:UserInstallation=file://{tmp}/profile", "--headless",
                   "--convert-to", "pdf", "--outdir", tmp, path]
        subprocess.run(command, check=True, capture_output=True, timeout=LIBREOFFICE_TIMEOUT_S)

        base_name = os.path.splitext(os.path.basename(path))[0]
        produced = os.path.join(tmp, base_name + ".pdf")
        if not os.path.exists(produced):
            raise RuntimeError("Conversion failed. Export the deck as PDF and open that instead.")
        shutil.move(produced, out)


def to_pdf(path: str) -> str:
    """Return a PDF path for any supported file (cached by content hash)."""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext}. Use one of: {', '.join(SUPPORTED)}")
    if ext == ".pdf":
        return path

    # Name the cached PDF after the file's contents, not its name, so an
    # edited deck is converted again but an unchanged one never is.
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{_file_hash(path)}.pdf")
    if os.path.exists(out):
        return out

    print(f"Converting {os.path.basename(path)} to PDF (first time only)...")
    if ext == ".key" and sys.platform == "darwin":
        _keynote_to_pdf(path, out)
    else:
        _libreoffice_to_pdf(path, out)
    return out


def load(path: str, zoom: float = RENDER_ZOOM) -> list[np.ndarray]:
    """Open a presentation and return one BGR image per slide."""
    # Imported here so the rest of the package can be used without PyMuPDF.
    import pymupdf

    slides = []
    with pymupdf.open(to_pdf(path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
            slides.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))   # OpenCV expects BGR
    if not slides:
        raise RuntimeError("The presentation has no slides.")
    return slides


def pick_file() -> str | None:
    """Native file dialog: AppleScript on macOS, tkinter elsewhere."""
    if sys.platform == "darwin":
        result = subprocess.run(["osascript", "-e", f'POSIX path of (choose file with prompt "{DIALOG_TITLE}")'],
                                capture_output=True, text=True)
        return result.stdout.strip() or None

    try:
        from tkinter import Tk, filedialog
    except ImportError:
        return None
    Tk().withdraw()     # hide the empty main window tkinter would show
    patterns = " ".join("*" + ext for ext in SUPPORTED)
    chosen = filedialog.askopenfilename(title=DIALOG_TITLE, filetypes=[("Presentations", patterns)])
    return chosen or None


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox img into a black w x h canvas."""
    img_h, img_w = img.shape[:2]
    # Scale by the tighter of the two ratios so the whole slide fits, unstretched.
    scale = min(w / img_w, h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    canvas = np.zeros((h, w, 3), np.uint8)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return canvas
