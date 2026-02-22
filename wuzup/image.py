"""Image loading, compositing, and OCR utilities."""

import logging
import os
import shutil
import sys
from pathlib import Path

import pytesseract
from PIL import Image

# When bundled via PyInstaller, Tesseract is included in the _MEIPASS directory.
_BUNDLED_TESSERACT_DIR = Path(getattr(sys, "_MEIPASS", ""), "tesseract")
_TESSERACT_BINARY = "tesseract.exe" if sys.platform == "win32" else "tesseract"
_FALLBACK_DIRS = [_BUNDLED_TESSERACT_DIR]
if sys.platform == "win32":
    _FALLBACK_DIRS.append(Path(r"C:\Program Files\Tesseract-OCR"))

if shutil.which("tesseract") is None:
    for _dir in _FALLBACK_DIRS:
        if (_dir / _TESSERACT_BINARY).is_file():
            os.environ["PATH"] = str(_dir) + os.pathsep + os.environ.get("PATH", "")
            break

log = logging.getLogger(__name__)


def load_image_from_path(path: str) -> Image.Image:
    """Open a local image file and return it as a PIL Image.

    Args:
        path: Filesystem path to the image file.

    Returns:
        The opened PIL Image.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    log.debug(f"Loading image from {path}")
    img = Image.open(path)
    log.debug(f"Loaded image: {img.size[0]}x{img.size[1]} {img.mode}")
    return img


def composite_on_color(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Composite an RGBA image onto a solid-color background.

    If the input image is not already RGBA it is converted first.  The
    alpha channel is used as the paste mask so that transparent regions
    reveal the background *color*.

    Args:
        image: Source image (any mode; will be converted to RGBA).
        color: RGB tuple used as the background fill color.

    Returns:
        A new RGB image with the source composited over the background.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    bg = Image.new("RGB", image.size, color)
    bg.paste(image, mask=image.split()[3])
    return bg


def ocr_variant(variant: Image.Image) -> str:
    """Run Tesseract OCR on a single image variant.

    Args:
        variant: A PIL Image to perform OCR on.

    Returns:
        The recognised text with leading/trailing whitespace stripped.
    """
    return pytesseract.image_to_string(variant).strip()


def image_to_text(image: Image.Image) -> str:
    """Extract text from an image using multiple OCR passes.

    The image is composited onto both a white and a black background and
    the individual R, G, and B channels of the white-background version
    are also extracted.  OCR is run on all five variants and the unique
    lines (compared case-insensitively) are collected in order.

    Args:
        image: Source image (typically RGBA or RGB).

    Returns:
        A newline-joined string of unique OCR-detected lines.
    """
    # Composite on both white and black backgrounds.
    # White bg: makes dark/colored text visible (normal case).
    # Black bg: makes light/white text visible (text that's invisible on white).
    on_white = composite_on_color(image, (255, 255, 255))
    on_black = composite_on_color(image, (0, 0, 0))

    # Per-channel extraction from the white-bg version to catch colored text
    # e.g. cyan text has R≈0 on white R=255 → high contrast in R channel
    r_ch = on_white.getchannel("R")
    g_ch = on_white.getchannel("G")
    b_ch = on_white.getchannel("B")

    variants = [on_white, on_black, r_ch, g_ch, b_ch]
    log.debug(f"Running OCR on {len(variants)} variant(s) of {image.size[0]}x{image.size[1]} image")

    # Collect unique lines across all variants
    seen: set[str] = set()
    all_lines: list[str] = []

    for v in variants:
        text = ocr_variant(v)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            key = line.lower()
            if key not in seen:
                seen.add(key)
                all_lines.append(line)

    log.debug(f"OCR extracted {len(all_lines)} unique line(s) from image")
    return "\n".join(all_lines)
