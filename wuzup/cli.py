"""Command-line interface for wuzup."""

import argparse
import io
import logging
import sys

from wuzup.fetcher import Fetcher
from wuzup.image import image_to_text

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

wuzup_logger = logging.getLogger("wuzup")
wuzup_logger.setLevel(logging.INFO)


def _debug_setup(debug: bool, debug_all: bool):
    """Configure log levels based on CLI debug flags.

    Args:
        debug: When ``True``, set the ``wuzup`` logger to DEBUG.
        debug_all: When ``True``, set the root logger to DEBUG so that
            all libraries emit debug output as well.
    """
    if debug:
        log.info("Turning on debug logging via --debug")
        wuzup_logger.setLevel(logging.DEBUG)
    if debug_all:
        log.info("Turning on ALL debug logging via --debug-all")
        logging.getLogger().setLevel(logging.DEBUG)
        wuzup_logger.setLevel(logging.DEBUG)


def _make_fetcher(use_playwright: bool, timeout: float = 30, page_wait_for_timeout: float = 0) -> Fetcher:
    """Instantiate the appropriate :class:`Fetcher` subclass.

    Import is deferred so that Playwright is only loaded when needed.

    Args:
        use_playwright: If ``True``, return a :class:`PlaywrightFetcher`.
        timeout: Timeout in seconds for HTTP requests.
        page_wait_for_timeout: Extra wait after page load (Playwright only).

    Returns:
        A :class:`Fetcher` instance.
    """
    if use_playwright:
        from wuzup.playwright_fetcher import PlaywrightFetcher

        log.debug(f"Creating PlaywrightFetcher (timeout={timeout}, page_wait_for_timeout={page_wait_for_timeout})")
        return PlaywrightFetcher(timeout=timeout, page_wait_for_timeout=page_wait_for_timeout)

    if page_wait_for_timeout:
        raise ValueError("--page-wait-for-timeout requires --playwright or --fallback-to-playwright")

    from wuzup.requests_fetcher import RequestsFetcher

    log.debug(f"Creating RequestsFetcher (timeout={timeout})")
    return RequestsFetcher(timeout=timeout)


def _scrape_to_text(
    fetcher: Fetcher,
    url: str,
    selectors: list[str],
) -> str | None:
    """Scrape *url* with *fetcher* and return extracted text, or ``None``.

    Matched images are OCR'd and combined with the visible text of
    matched elements.  Returns ``None`` when neither images nor text
    are found (the "empty" case that may trigger a Playwright fallback).

    Args:
        fetcher: The :class:`Fetcher` to use.
        url: Page URL.
        selectors: CSS selectors to locate elements.

    Returns:
        Extracted text, or ``None`` if nothing was found.
    """
    log.debug(f"Scraping {url} with {len(selectors)} selector(s): {selectors}")
    result = fetcher.scrape(url, selectors)
    log.debug(f"Scrape found {len(result.img_urls)} image URL(s) and {len(result.element_text)} chars of text")
    images = fetcher.fetch_images(result.img_urls)

    parts: list[str] = []

    if images:
        log.debug(f"Running OCR on {len(images)} fetched image(s)")
        all_lines: list[str] = []
        seen: set[str] = set()
        for image in images:
            text = image_to_text(image)
            for line in text.splitlines():
                key = line.lower()
                if key not in seen:
                    seen.add(key)
                    all_lines.append(line)
        log.debug(f"OCR produced {len(all_lines)} unique line(s)")
        parts.append("\n".join(all_lines))

    if result.element_text.strip():
        log.debug("Including element text in output")
        parts.append(result.element_text)

    if parts:
        return "\n".join(parts)

    return None


def _web_to_text_command(
    url: str,
    selectors: list[str] | None = None,
    timeout: float = 30,
    page_wait_for_timeout: float = 0,
    use_playwright: bool = False,
    fallback_to_playwright: bool = False,
) -> str:
    """Fetch images and text from a web page via selectors.

    OCR text from matched images and visible element text are combined.
    If neither images nor text are found a ``ValueError`` is raised.

    When *fallback_to_playwright* is ``True`` and the initial requests-based
    fetch finds no data, the page is re-fetched using Playwright before
    raising.  Connection errors and other exceptions are **not** caught
    by the fallback — only the "no data matched" case triggers it.

    Args:
        url: URL pointing to an image or an HTML page containing one.
        selectors: Optional list of CSS selectors to locate ``<img>``
            elements on the page.  Images matching *any* selector are
            processed.  When ``None`` the *url* is fetched as a direct
            image link.
        timeout: Timeout in seconds for HTTP requests.
        page_wait_for_timeout: Extra time in seconds to wait after page
            load for JavaScript redirects.
        use_playwright: If ``True``, use Playwright to render the page
            (evaluates JavaScript).  Defaults to ``False``.
        fallback_to_playwright: If ``True``, try with plain requests
            first and automatically retry with Playwright when no data
            is found.  Mutually exclusive with *use_playwright*.

    Returns:
        Combined OCR text from matched images and visible element text.
    """
    log.debug(f"web-to-text called with url={url}, selectors={selectors}")

    if page_wait_for_timeout and not use_playwright and not fallback_to_playwright:
        raise ValueError("--page-wait-for-timeout requires --playwright or --fallback-to-playwright")

    fetcher = _make_fetcher(
        use_playwright,
        timeout=timeout,
        page_wait_for_timeout=page_wait_for_timeout if use_playwright else 0,
    )

    if not selectors:
        log.debug(f"No selectors given; treating URL as direct image: {url}")
        image = fetcher.fetch_image(url)
        return image_to_text(image)

    text = _scrape_to_text(fetcher, url, selectors)
    if text is not None:
        return text

    # Nothing found - try Playwright fallback if requested.
    if fallback_to_playwright and not use_playwright:
        log.info("No data found via requests; falling back to Playwright")
        pw_fetcher = _make_fetcher(True, timeout=timeout, page_wait_for_timeout=page_wait_for_timeout)
        text = _scrape_to_text(pw_fetcher, url, selectors)
        if text is not None:
            return text

    raise ValueError("No images or text found matching selectors")


def main(args: list[str] | None = None, output: io.TextIOBase | None = None):
    """Entry point for the ``wuzup`` CLI.

    Args:
        args: Optional list of command-line arguments. If ``None``,
            ``sys.argv`` is used (via ``argparse`` default behavior).
        output: Optional writable text stream. If ``None``, defaults
            to ``sys.stdout``.
    """
    if output is None:
        output = sys.stdout
    parser = argparse.ArgumentParser(description="wuzup - a multi-tool CLI")
    parser.add_argument("--debug", action="store_true", help="If given, turn on wuzup debug logging.")
    parser.add_argument("--debug-all", action="store_true", help="If given, turn on ALL debug logging globally.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── web-to-text ─────────────────────────────────────────────────
    wtt_parser = subparsers.add_parser(
        "web-to-text",
        aliases=["w2t", "wtt"],
        help="Extract text from a web page: OCR matched images first, fall back to element text.",
    )
    wtt_parser.add_argument("-u", "--url", type=str, required=True, help="URL to a page (or direct image).")
    wtt_parser.add_argument(
        "-s",
        "--selector",
        type=str,
        action="append",
        dest="selectors",
        help="CSS selector to find images/elements on the page. Can be specified multiple times.",
    )
    wtt_parser.add_argument(
        "-T",
        "--timeout",
        type=float,
        default=30,
        help="Timeout in seconds for HTTP requests (default: 30).",
    )
    wtt_parser.add_argument(
        "--page-wait-for-timeout",
        type=float,
        default=0,
        help="Extra time in seconds to wait after page load for JS redirects (default: 0). Only allowed if --playwright is used.",
    )
    pw_group = wtt_parser.add_mutually_exclusive_group()
    pw_group.add_argument(
        "--playwright",
        action="store_true",
        default=False,
        help="Use Playwright (headless Chromium) to render the page with JavaScript. Without this flag, plain HTTP requests are used.",
    )
    pw_group.add_argument(
        "-F",
        "--fallback-to-playwright",
        action="store_true",
        default=False,
        help="Use plain requests first; if no data is found, retry with Playwright. Mutually exclusive with --playwright.",
    )

    args = parser.parse_args(args)
    _debug_setup(args.debug, args.debug_all)

    if args.command in ("web-to-text", "w2t", "wtt"):
        print(
            _web_to_text_command(
                url=args.url,
                selectors=args.selectors,
                timeout=args.timeout,
                page_wait_for_timeout=args.page_wait_for_timeout,
                use_playwright=args.playwright,
                fallback_to_playwright=args.fallback_to_playwright,
            ),
            file=output,
        )


if __name__ == "__main__":
    main()
