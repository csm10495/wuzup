"""Fetcher implementation using ``requests`` + BeautifulSoup (no JavaScript)."""

from __future__ import annotations

import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from wuzup.fetcher import Fetcher, ScrapeResult

log = logging.getLogger(__name__)


class RequestsFetcher(Fetcher):
    """Fetch pages with ``requests`` and parse with BeautifulSoup.

    No JavaScript is evaluated—only the raw HTML returned by the server
    is inspected.

    Args:
        timeout: Default timeout in seconds for HTTP requests.
    """

    def scrape(self, url: str, selectors: list[str]) -> ScrapeResult:
        """Fetch *url* and extract image URLs + text from *selectors*.

        Args:
            url: URL of the HTML page to fetch.
            selectors: CSS selectors used to locate elements.

        Returns:
            A :class:`ScrapeResult` with image URLs and element text.
        """
        log.debug(f"Fetching page {url} via requests (timeout={self.timeout}s)")
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        log.debug(f"Got HTTP {resp.status_code} ({len(resp.text)} chars)")
        soup = BeautifulSoup(resp.text, "html.parser")

        img_urls: list[str] = []
        seen_urls: set[str] = set()
        text_parts: list[str] = []

        def _record_img(tag, selector: str) -> None:
            img_url = tag.get("src") or tag.get("data-src")
            if img_url:
                resolved = urljoin(url, img_url)
                if resolved not in seen_urls:
                    seen_urls.add(resolved)
                    img_urls.append(resolved)
                    log.debug(f"Found image URL: {resolved}")
            else:
                log.debug(f"Element matching selector '{selector}' has no src or data-src attribute, skipping")

        for selector in selectors:
            for tag in soup.select(selector):
                _record_img(tag, selector)
                for nested_img in tag.find_all("img"):
                    _record_img(nested_img, f"{selector} img")

                text = tag.get_text(separator=" ", strip=True)
                if text:
                    text_parts.append(text)
                    log.debug(f"Extracted text from selector '{selector}': {text[:80]}...")

        log.debug(f"RequestsFetcher scrape complete: {len(img_urls)} image(s), {len(text_parts)} text block(s)")
        return ScrapeResult(img_urls=img_urls, element_text="\n".join(text_parts))
