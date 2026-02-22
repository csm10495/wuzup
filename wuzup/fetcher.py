"""Abstract base for page fetchers and shared data types."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO

import cairosvg
import requests
from PIL import Image

log = logging.getLogger(__name__)


def _is_svg(data: bytes) -> bool:
    """Return ``True`` if *data* looks like an SVG image."""
    header = data[:512].lstrip()
    return header.startswith(b"<svg") or (header.startswith(b"<?xml") and b"<svg" in header[:1024])


def _svg_to_pil(data: bytes) -> Image.Image:
    """Render SVG *data* to a PIL Image via CairoSVG."""
    png_data = cairosvg.svg2png(bytestring=data)
    return Image.open(BytesIO(png_data))


@dataclass
class ScrapeResult:
    """Container for data extracted from a web page via selectors.

    Attributes:
        img_urls: Deduplicated absolute URLs of images found via selectors.
        element_text: Visible text extracted from matched selector elements,
            joined by newlines.
    """

    img_urls: list[str] = field(default_factory=list)
    element_text: str = ""


class Fetcher(ABC):
    """Base class for page fetchers.

    Subclasses implement :meth:`scrape` to load a page and extract both
    image URLs and element text in a single pass.

    Args:
        timeout: Default timeout in seconds for HTTP requests.
    """

    def __init__(self, timeout: float = 30) -> None:
        self.timeout = timeout

    def fetch_image(self, url: str) -> Image.Image:
        """Fetch an image directly from *url* via HTTP GET.

        Args:
            url: URL pointing to an image.

        Returns:
            The fetched PIL Image.

        Raises:
            requests.HTTPError: If the HTTP request fails.
        """
        log.debug(f"Fetching image from {url} (timeout={self.timeout}s)")
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        if _is_svg(resp.content):
            log.debug("Detected SVG content, converting to PNG")
            image = _svg_to_pil(resp.content)
        else:
            image = Image.open(BytesIO(resp.content))
        log.debug(f"Fetched image: {image.size[0]}x{image.size[1]} {image.mode}")
        return image

    def fetch_images(self, img_urls: list[str]) -> list[Image.Image]:
        """Download each URL in *img_urls*, skipping failures.

        Args:
            img_urls: Absolute image URLs.

        Returns:
            A list of successfully fetched PIL Images (may be empty).
        """
        log.debug(f"Fetching {len(img_urls)} image(s)")
        images: list[Image.Image] = []
        for img_url in img_urls:
            try:
                images.append(self.fetch_image(img_url))
            except Exception:
                log.debug(f"Failed to fetch image at {img_url}", exc_info=True)
        log.debug(f"Successfully fetched {len(images)}/{len(img_urls)} image(s)")
        return images

    @abstractmethod
    def scrape(self, url: str, selectors: list[str]) -> ScrapeResult:
        """Load the page at *url* **once** and return images + text.

        Implementations must inspect elements matching *selectors* for
        ``src`` / ``data-src`` image attributes **and** collect visible
        text from those same elements in a single pass.

        Args:
            url: URL of the HTML page.
            selectors: CSS selectors used to locate elements.

        Returns:
            A :class:`ScrapeResult` containing image URLs and element text.
        """
