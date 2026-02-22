"""Fetcher implementation using Playwright (headless Chromium, evaluates JS)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

from wuzup.fetcher import Fetcher, ScrapeResult

log = logging.getLogger(__name__)


@contextmanager
def _get_page():
    """Launch a headless Chromium browser and yield a Page.

    The browser is automatically closed when the context manager exits.

    Yields:
        A Playwright Page instance.
    """
    with sync_playwright() as p:
        launch_kwargs: dict = {"headless": True}
        all_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
        if all_proxy:
            log.debug(f"Using proxy: {all_proxy}")
            launch_kwargs["proxy"] = {"server": all_proxy}
        log.debug("Launching headless Chromium")
        browser = p.chromium.launch(**launch_kwargs)
        page = browser.new_page()
        try:
            yield page
        finally:
            browser.close()


class PlaywrightFetcher(Fetcher):
    """Fetch pages with a headless Chromium browser via Playwright.

    JavaScript is fully evaluated before selectors are inspected.

    Args:
        timeout: Default timeout in seconds for HTTP requests.
        page_wait_for_timeout: Extra seconds to wait after page load
            (e.g. for JavaScript redirects).  Defaults to ``0``.
    """

    def __init__(self, timeout: float = 30, page_wait_for_timeout: float = 0) -> None:
        super().__init__(timeout=timeout)
        self.page_wait_for_timeout = page_wait_for_timeout

    def scrape(self, url: str, selectors: list[str]) -> ScrapeResult:
        """Render *url* in Chromium and extract image URLs + text from *selectors*.

        The page is loaded **once**; both image URLs and element text are
        collected in a single pass.

        Args:
            url: URL of the HTML page to render.
            selectors: CSS selectors used to locate elements.

        Returns:
            A :class:`ScrapeResult` with image URLs and element text.

        Raises:
            RuntimeError: If the page returns a non-OK HTTP status.
        """
        with _get_page() as page:
            log.debug(f"Navigating to {url} (timeout={self.timeout}s)")
            response = page.goto(url, timeout=self.timeout * 1000, wait_until="load")

            if self.page_wait_for_timeout > 0:
                log.debug(f"Waiting additional {self.page_wait_for_timeout}s after page load")
                page.wait_for_timeout(self.page_wait_for_timeout * 1000)
            if response is not None:
                log.debug(f"Page response: HTTP {response.status}")
            if response is not None and not response.ok:
                raise RuntimeError(f"HTTP {response.status}: {response.status_text} for {url}")

            img_urls: list[str] = []
            seen_urls: set[str] = set()
            text_parts: list[str] = []

            def _collect_img(element, selector: str) -> None:
                img_url = element.get_attribute("src") or element.get_attribute("data-src")
                if img_url:
                    resolved = urljoin(url, img_url)
                    if resolved not in seen_urls:
                        seen_urls.add(resolved)
                        img_urls.append(resolved)
                        log.debug(f"Found image URL: {resolved}")
                else:
                    log.debug(f"Element matching selector '{selector}' has no src or data-src attribute, skipping")

            for selector in selectors:
                for element in page.query_selector_all(selector):
                    _collect_img(element, selector)
                    # Also look for <img> tags nested inside the matched element
                    for nested_img in element.query_selector_all("img"):
                        _collect_img(nested_img, f"{selector} img")

                    text = (element.inner_text() or "").strip()
                    if text:
                        text_parts.append(text)
                        log.debug(f"Extracted text from selector '{selector}': {text[:80]}...")

            log.debug(f"PlaywrightFetcher scrape complete: {len(img_urls)} image(s), {len(text_parts)} text block(s)")
            return ScrapeResult(img_urls=img_urls, element_text="\n".join(text_parts))
