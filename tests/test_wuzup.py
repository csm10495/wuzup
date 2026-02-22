"""Comprehensive unit and integration tests for wuzup."""

import logging
import os
import subprocess
import sys
from io import BytesIO, StringIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from wuzup.cli import _debug_setup, _make_fetcher, _web_to_text_command, main
from wuzup.fetcher import Fetcher, ScrapeResult, _is_svg, _svg_to_pil
from wuzup.image import composite_on_color, image_to_text, load_image_from_path, ocr_variant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgba_image(size=(10, 10), color=(255, 0, 0, 128)) -> Image.Image:
    """Create a small RGBA test image."""
    return Image.new("RGBA", size, color)


def _make_rgb_image(size=(10, 10), color=(255, 0, 0)) -> Image.Image:
    """Create a small RGB test image."""
    return Image.new("RGB", size, color)


def _image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _mock_page_cm(mock_page):
    """Create a mock context manager that yields *mock_page*."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield mock_page

    return _cm


# ===========================================================================
# fetcher.py — _is_svg / _svg_to_pil
# ===========================================================================


class TestIsSvg:
    def test_bare_svg_tag(self):
        assert _is_svg(b"<svg><rect/></svg>") is True

    def test_xml_declaration_with_svg(self):
        assert _is_svg(b'<?xml version="1.0"?><svg><circle/></svg>') is True

    def test_leading_whitespace(self):
        assert _is_svg(b"   \n  <svg></svg>") is True

    def test_png_bytes(self):
        img_bytes = _image_to_bytes(_make_rgb_image())
        assert _is_svg(img_bytes) is False

    def test_empty_bytes(self):
        assert _is_svg(b"") is False

    def test_plain_xml_without_svg(self):
        assert _is_svg(b'<?xml version="1.0"?><html></html>') is False


class TestSvgToPil:
    def test_converts_simple_svg(self):
        svg_data = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>'
        result = _svg_to_pil(svg_data)
        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)

    @patch("wuzup.fetcher.cairosvg", create=True)
    def test_calls_cairosvg_svg2png(self, mock_cairosvg):
        """Verify _svg_to_pil delegates to cairosvg.svg2png."""
        png_img = _make_rgb_image()
        mock_cairosvg.svg2png.return_value = _image_to_bytes(png_img)

        # We need to patch at the point of import inside the function
        with patch("wuzup.fetcher.cairosvg", mock_cairosvg, create=True):
            # Since cairosvg is lazily imported, we need to mock the import
            import wuzup.fetcher as fetcher_mod

            with patch.dict("sys.modules", {"cairosvg": mock_cairosvg}):
                result = fetcher_mod._svg_to_pil(b"<svg></svg>")

        mock_cairosvg.svg2png.assert_called_once_with(bytestring=b"<svg></svg>")
        assert isinstance(result, Image.Image)


# ===========================================================================
# fetcher.py — ScrapeResult
# ===========================================================================


class TestScrapeResult:
    def test_defaults(self):
        result = ScrapeResult()
        assert result.img_urls == []
        assert result.element_text == ""

    def test_custom_values(self):
        result = ScrapeResult(img_urls=["http://a.png"], element_text="hello")
        assert result.img_urls == ["http://a.png"]
        assert result.element_text == "hello"

    def test_independent_instances(self):
        """Each instance gets its own list."""
        r1 = ScrapeResult()
        r2 = ScrapeResult()
        r1.img_urls.append("x")
        assert r2.img_urls == []


# ===========================================================================
# fetcher.py — Fetcher (abstract base, concrete helpers)
# ===========================================================================


class _ConcreteFetcher(Fetcher):
    """Minimal concrete subclass for testing base methods."""

    def scrape(self, url, selectors):
        return ScrapeResult()


class TestFetcher:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Fetcher()

    def test_timeout_default(self):
        f = _ConcreteFetcher()
        assert f.timeout == 30

    def test_timeout_custom(self):
        f = _ConcreteFetcher(timeout=10)
        assert f.timeout == 10

    @patch("wuzup.fetcher.requests.get")
    def test_fetch_image(self, mock_get):
        img = _make_rgb_image()
        mock_resp = MagicMock()
        mock_resp.content = _image_to_bytes(img)
        mock_get.return_value = mock_resp

        f = _ConcreteFetcher(timeout=5)
        result = f.fetch_image("http://example.com/img.png")

        mock_get.assert_called_once_with("http://example.com/img.png", timeout=5)
        mock_resp.raise_for_status.assert_called_once()
        assert isinstance(result, Image.Image)

    @patch("wuzup.fetcher._svg_to_pil")
    @patch("wuzup.fetcher.requests.get")
    def test_fetch_image_svg_content(self, mock_get, mock_svg_to_pil):
        """SVG responses are routed through _svg_to_pil."""
        svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        mock_resp = MagicMock()
        mock_resp.content = svg_data
        mock_get.return_value = mock_resp
        expected_img = _make_rgb_image()
        mock_svg_to_pil.return_value = expected_img

        f = _ConcreteFetcher()
        result = f.fetch_image("http://example.com/icon.svg")

        mock_svg_to_pil.assert_called_once_with(svg_data)
        assert result is expected_img

    @patch("wuzup.fetcher.requests.get")
    def test_fetch_images_skips_failures(self, mock_get):
        """If one URL fails, the other images are still returned."""
        good_img = _make_rgb_image()
        good_resp = MagicMock()
        good_resp.content = _image_to_bytes(good_img)

        def side_effect(url, **kw):
            if "bad" in url:
                raise ConnectionError("boom")
            return good_resp

        mock_get.side_effect = side_effect

        f = _ConcreteFetcher()
        imgs = f.fetch_images(["http://bad.com/x.png", "http://good.com/y.png"])

        assert len(imgs) == 1
        assert isinstance(imgs[0], Image.Image)

    @patch("wuzup.fetcher.requests.get")
    def test_fetch_images_empty_list(self, mock_get):
        f = _ConcreteFetcher()
        assert f.fetch_images([]) == []
        mock_get.assert_not_called()


# ===========================================================================
# requests_fetcher.py — RequestsFetcher
# ===========================================================================


class TestRequestsFetcher:
    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_finds_src(self, mock_get):
        """img tags with src are collected."""
        from wuzup.requests_fetcher import RequestsFetcher

        html = '<html><body><img class="t" src="/a.png"></body></html>'
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".t"])

        assert result.img_urls == ["http://example.com/a.png"]

    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_finds_data_src(self, mock_get):
        from wuzup.requests_fetcher import RequestsFetcher

        html = '<html><body><div class="t" data-src="/lazy.jpg"></div></body></html>'
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".t"])
        assert result.img_urls == ["http://example.com/lazy.jpg"]

    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_nested_img(self, mock_get):
        """img nested inside a matching div is found."""
        from wuzup.requests_fetcher import RequestsFetcher

        html = '<html><body><div class="wrap"><img src="/nested.png"></div></body></html>'
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".wrap"])
        assert "http://example.com/nested.png" in result.img_urls

    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_deduplication(self, mock_get):
        """Same image URL appearing twice is deduplicated."""
        from wuzup.requests_fetcher import RequestsFetcher

        html = '<html><body><img class="t" src="/a.png"><img class="t" src="/a.png"></body></html>'
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".t"])
        assert result.img_urls == ["http://example.com/a.png"]

    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_text_extraction(self, mock_get):
        from wuzup.requests_fetcher import RequestsFetcher

        html = '<html><body><p class="t">Hello World</p></body></html>'
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".t"])
        assert result.element_text == "Hello World"

    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_multiple_selectors(self, mock_get):
        from wuzup.requests_fetcher import RequestsFetcher

        html = (
            '<html><body><img class="a" src="/1.png"><div class="b"><img src="/2.png"><p>text</p></div></body></html>'
        )
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".a", ".b"])
        assert len(result.img_urls) == 2
        assert "text" in result.element_text

    @patch("wuzup.requests_fetcher.requests.get")
    def test_scrape_no_match(self, mock_get):
        from wuzup.requests_fetcher import RequestsFetcher

        html = "<html><body><p>nothing</p></body></html>"
        mock_get.return_value = MagicMock(text=html)

        f = RequestsFetcher()
        result = f.scrape("http://example.com", [".missing"])
        assert result.img_urls == []
        assert result.element_text == ""


# ===========================================================================
# playwright_fetcher.py — _get_page / PlaywrightFetcher
# ===========================================================================


class TestGetPage:
    @patch("wuzup.playwright_fetcher.sync_playwright")
    def test_yields_page_and_closes(self, mock_sp):
        from wuzup.playwright_fetcher import _get_page

        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_browser.new_page.return_value = mock_page
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_sp.return_value.__enter__ = MagicMock(return_value=mock_pw)
        mock_sp.return_value.__exit__ = MagicMock(return_value=False)

        with _get_page() as page:
            assert page is mock_page

        mock_browser.close.assert_called_once()

    @patch.dict(os.environ, {"ALL_PROXY": "http://proxy:8080"}, clear=False)
    @patch("wuzup.playwright_fetcher.sync_playwright")
    def test_uses_proxy_from_env(self, mock_sp):
        from wuzup.playwright_fetcher import _get_page

        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_browser.new_page.return_value = mock_page
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_sp.return_value.__enter__ = MagicMock(return_value=mock_pw)
        mock_sp.return_value.__exit__ = MagicMock(return_value=False)

        with _get_page() as _page:
            pass

        launch_kwargs = mock_pw.chromium.launch.call_args
        assert launch_kwargs[1].get("proxy") == {"server": "http://proxy:8080"}

    @patch("wuzup.playwright_fetcher.sync_playwright")
    def test_closes_browser_on_error(self, mock_sp):
        from wuzup.playwright_fetcher import _get_page

        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_browser.new_page.return_value = mock_page
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_sp.return_value.__enter__ = MagicMock(return_value=mock_pw)
        mock_sp.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(RuntimeError):
            with _get_page() as _page:
                raise RuntimeError("test")

        mock_browser.close.assert_called_once()


class TestPlaywrightFetcher:
    def _make_fetcher_and_page(self):
        from wuzup.playwright_fetcher import PlaywrightFetcher

        mock_page = MagicMock()
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status = 200
        mock_page.goto.return_value = mock_response

        fetcher = PlaywrightFetcher(timeout=10, page_wait_for_timeout=0)
        return fetcher, mock_page

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_collects_src(self, mock_gp):
        fetcher, mock_page = self._make_fetcher_and_page()
        mock_gp.return_value = _mock_page_cm(mock_page)()

        elem = MagicMock()
        elem.get_attribute.side_effect = lambda a: "/img.png" if a == "src" else None
        elem.query_selector_all.return_value = []
        elem.inner_text.return_value = ""
        mock_page.query_selector_all.return_value = [elem]

        result = fetcher.scrape("http://example.com", [".sel"])
        assert result.img_urls == ["http://example.com/img.png"]

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_nested_img(self, mock_gp):
        fetcher, mock_page = self._make_fetcher_and_page()
        mock_gp.return_value = _mock_page_cm(mock_page)()

        container = MagicMock()
        container.get_attribute.return_value = None
        container.inner_text.return_value = "some text"

        nested = MagicMock()
        nested.get_attribute.side_effect = lambda a: "/nested.png" if a == "src" else None
        container.query_selector_all.return_value = [nested]

        mock_page.query_selector_all.return_value = [container]

        result = fetcher.scrape("http://example.com", [".wrap"])
        assert "http://example.com/nested.png" in result.img_urls
        assert result.element_text == "some text"

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_text_extraction(self, mock_gp):
        fetcher, mock_page = self._make_fetcher_and_page()
        mock_gp.return_value = _mock_page_cm(mock_page)()

        elem = MagicMock()
        elem.get_attribute.return_value = None
        elem.query_selector_all.return_value = []
        elem.inner_text.return_value = "Hello World"
        mock_page.query_selector_all.return_value = [elem]

        result = fetcher.scrape("http://example.com", [".sel"])
        assert result.element_text == "Hello World"

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_deduplication(self, mock_gp):
        fetcher, mock_page = self._make_fetcher_and_page()
        mock_gp.return_value = _mock_page_cm(mock_page)()

        elem1 = MagicMock()
        elem1.get_attribute.side_effect = lambda a: "/same.png" if a == "src" else None
        elem1.query_selector_all.return_value = []
        elem1.inner_text.return_value = ""

        elem2 = MagicMock()
        elem2.get_attribute.side_effect = lambda a: "/same.png" if a == "src" else None
        elem2.query_selector_all.return_value = []
        elem2.inner_text.return_value = ""

        mock_page.query_selector_all.return_value = [elem1, elem2]

        result = fetcher.scrape("http://example.com", [".sel"])
        assert result.img_urls == ["http://example.com/same.png"]

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_http_error_raises(self, mock_gp):
        fetcher, mock_page = self._make_fetcher_and_page()
        mock_gp.return_value = _mock_page_cm(mock_page)()

        mock_resp = mock_page.goto.return_value
        mock_resp.ok = False
        mock_resp.status = 404
        mock_resp.status_text = "Not Found"

        with pytest.raises(RuntimeError, match="HTTP 404"):
            fetcher.scrape("http://example.com/bad", [".sel"])

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_page_wait_for_timeout(self, mock_gp):
        from wuzup.playwright_fetcher import PlaywrightFetcher

        mock_page = MagicMock()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_page.goto.return_value = mock_resp
        mock_page.query_selector_all.return_value = []
        mock_gp.return_value = _mock_page_cm(mock_page)()

        fetcher = PlaywrightFetcher(timeout=10, page_wait_for_timeout=2)
        fetcher.scrape("http://example.com", [".sel"])

        mock_page.wait_for_timeout.assert_called_once_with(2000)

    @patch("wuzup.playwright_fetcher._get_page")
    def test_scrape_no_wait_when_zero(self, mock_gp):
        fetcher, mock_page = self._make_fetcher_and_page()
        mock_gp.return_value = _mock_page_cm(mock_page)()
        mock_page.query_selector_all.return_value = []

        fetcher.scrape("http://example.com", [".sel"])
        mock_page.wait_for_timeout.assert_not_called()


# ===========================================================================
# image.py — load_image_from_path
# ===========================================================================


class TestLoadImageFromPath:
    def test_load_valid_image(self, tmp_path):
        p = tmp_path / "test.png"
        _make_rgb_image().save(str(p))
        img = load_image_from_path(str(p))
        assert isinstance(img, Image.Image)
        assert img.size == (10, 10)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_image_from_path("/nonexistent/image.png")


# ===========================================================================
# image.py — composite_on_color
# ===========================================================================


class TestCompositeOnColor:
    def test_rgba_on_white(self):
        img = _make_rgba_image(color=(255, 0, 0, 128))
        result = composite_on_color(img, (255, 255, 255))
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_rgb_converted_to_rgba_then_composited(self):
        img = _make_rgb_image(color=(0, 255, 0))
        result = composite_on_color(img, (0, 0, 0))
        assert result.mode == "RGB"

    def test_transparent_shows_bg(self):
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        result = composite_on_color(img, (123, 45, 67))
        assert result.getpixel((0, 0)) == (123, 45, 67)

    def test_opaque_hides_bg(self):
        img = Image.new("RGBA", (1, 1), (200, 100, 50, 255))
        result = composite_on_color(img, (0, 0, 0))
        assert result.getpixel((0, 0)) == (200, 100, 50)


# ===========================================================================
# image.py — ocr_variant
# ===========================================================================


class TestOcrVariant:
    @patch("wuzup.image.pytesseract.image_to_string", return_value="  hello world  ")
    def test_strips_whitespace(self, mock_ocr):
        result = ocr_variant(_make_rgb_image())
        assert result == "hello world"
        mock_ocr.assert_called_once()


# ===========================================================================
# image.py — image_to_text
# ===========================================================================


class TestImageToText:
    @patch("wuzup.image.ocr_variant")
    def test_deduplicates_case_insensitively(self, mock_ocr):
        mock_ocr.side_effect = ["Hello\nWorld", "hello\nworld", "HELLO", "World\nNew", "new"]
        result = image_to_text(_make_rgba_image())
        lines = result.splitlines()
        assert "Hello" in lines
        assert "World" in lines
        assert "New" in lines
        # Only unique lines (case-insensitive)
        lower_lines = [line.lower() for line in lines]
        assert len(lower_lines) == len(set(lower_lines))

    @patch("wuzup.image.ocr_variant", return_value="")
    def test_empty_ocr(self, mock_ocr):
        result = image_to_text(_make_rgba_image())
        assert result == ""

    @patch("wuzup.image.ocr_variant")
    def test_five_variants_called(self, mock_ocr):
        mock_ocr.return_value = ""
        image_to_text(_make_rgba_image())
        assert mock_ocr.call_count == 5


# ===========================================================================
# cli.py — _debug_setup
# ===========================================================================


class TestDebugSetup:
    def test_debug_sets_wuzup_to_debug(self):
        logger = logging.getLogger("wuzup")
        old = logger.level
        try:
            _debug_setup(debug=True, debug_all=False)
            assert logger.level == logging.DEBUG
        finally:
            logger.setLevel(old)

    def test_debug_all_sets_root_to_debug(self):
        root = logging.getLogger()
        old = root.level
        try:
            _debug_setup(debug=False, debug_all=True)
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(old)

    def test_no_debug(self):
        logger = logging.getLogger("wuzup")
        logger.setLevel(logging.INFO)
        _debug_setup(debug=False, debug_all=False)
        assert logger.level == logging.INFO


# ===========================================================================
# cli.py — _make_fetcher
# ===========================================================================


class TestMakeFetcher:
    def test_returns_requests_fetcher(self):
        from wuzup.requests_fetcher import RequestsFetcher

        f = _make_fetcher(use_playwright=False)
        assert isinstance(f, RequestsFetcher)
        assert f.timeout == 30

    def test_returns_playwright_fetcher(self):
        from wuzup.playwright_fetcher import PlaywrightFetcher

        f = _make_fetcher(use_playwright=True, timeout=15, page_wait_for_timeout=3)
        assert isinstance(f, PlaywrightFetcher)
        assert f.timeout == 15
        assert f.page_wait_for_timeout == 3

    def test_page_wait_without_playwright_raises(self):
        with pytest.raises(
            ValueError, match="--page-wait-for-timeout requires --playwright or --fallback-to-playwright"
        ):
            _make_fetcher(use_playwright=False, page_wait_for_timeout=5)

    def test_custom_timeout_requests(self):
        f = _make_fetcher(use_playwright=False, timeout=99)
        assert f.timeout == 99


# ===========================================================================
# cli.py — _web_to_text_command
# ===========================================================================


class TestWebToTextCommand:
    @patch("wuzup.cli.image_to_text", return_value="ocr text")
    @patch("wuzup.cli._make_fetcher")
    def test_direct_url_no_selectors(self, mock_mf, mock_i2t):
        """When no selectors are given, fetch_image is called directly."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_image.return_value = _make_rgb_image()
        mock_mf.return_value = mock_fetcher

        result = _web_to_text_command("http://example.com/pic.png")
        mock_fetcher.fetch_image.assert_called_once_with("http://example.com/pic.png")
        assert result == "ocr text"

    @patch("wuzup.cli.image_to_text", return_value="img text")
    @patch("wuzup.cli._make_fetcher")
    def test_selectors_with_images(self, mock_mf, mock_i2t):
        """When selectors find images, OCR the images."""
        mock_fetcher = MagicMock()
        mock_fetcher.scrape.return_value = ScrapeResult(img_urls=["http://example.com/img.png"])
        mock_fetcher.fetch_images.return_value = [_make_rgb_image()]
        mock_mf.return_value = mock_fetcher

        result = _web_to_text_command("http://example.com", selectors=[".sel"])
        mock_fetcher.scrape.assert_called_once_with("http://example.com", [".sel"])
        assert result == "img text"

    @patch("wuzup.cli._make_fetcher")
    def test_selectors_text_only(self, mock_mf):
        """When selectors find no images, return element text."""
        mock_fetcher = MagicMock()
        mock_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="fallback text")
        mock_fetcher.fetch_images.return_value = []
        mock_mf.return_value = mock_fetcher

        result = _web_to_text_command("http://example.com", selectors=[".sel"])
        assert result == "fallback text"

    @patch("wuzup.cli.image_to_text", return_value="ocr line")
    @patch("wuzup.cli._make_fetcher")
    def test_selectors_images_and_text_combined(self, mock_mf, mock_i2t):
        """When selectors find both images and text, combine them."""
        mock_fetcher = MagicMock()
        mock_fetcher.scrape.return_value = ScrapeResult(
            img_urls=["http://example.com/img.png"], element_text="element text"
        )
        mock_fetcher.fetch_images.return_value = [_make_rgb_image()]
        mock_mf.return_value = mock_fetcher

        result = _web_to_text_command("http://example.com", selectors=[".sel"])
        assert "ocr line" in result
        assert "element text" in result

    @patch("wuzup.cli._make_fetcher")
    def test_selectors_no_images_no_text_raises(self, mock_mf):
        """ValueError when neither images nor text are found."""
        mock_fetcher = MagicMock()
        mock_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="")
        mock_fetcher.fetch_images.return_value = []
        mock_mf.return_value = mock_fetcher

        with pytest.raises(ValueError, match="No images or text found"):
            _web_to_text_command("http://example.com", selectors=[".sel"])

    @patch("wuzup.cli.image_to_text")
    @patch("wuzup.cli._make_fetcher")
    def test_multiple_images_deduplicated(self, mock_mf, mock_i2t):
        """Lines from multiple images are deduplicated case-insensitively."""
        mock_fetcher = MagicMock()
        mock_fetcher.scrape.return_value = ScrapeResult(img_urls=["u1", "u2"])
        mock_fetcher.fetch_images.return_value = [_make_rgb_image(), _make_rgb_image()]
        mock_mf.return_value = mock_fetcher

        mock_i2t.side_effect = ["Line A\nLine B", "line a\nLine C"]

        result = _web_to_text_command("http://example.com", selectors=[".s"])
        lines = result.splitlines()
        assert "Line A" in lines
        assert "Line B" in lines
        assert "Line C" in lines
        lower_lines = [line.lower() for line in lines]
        assert len(lower_lines) == len(set(lower_lines))

    @patch("wuzup.cli.image_to_text", return_value="text")
    @patch("wuzup.cli._make_fetcher")
    def test_passes_playwright_flag(self, mock_mf, mock_i2t):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_image.return_value = _make_rgb_image()
        mock_mf.return_value = mock_fetcher

        _web_to_text_command("http://x.com/img.png", use_playwright=True, timeout=15, page_wait_for_timeout=2)
        mock_mf.assert_called_once_with(True, timeout=15, page_wait_for_timeout=2)

    @patch("wuzup.cli._make_fetcher")
    def test_fallback_to_playwright_triggers(self, mock_mf):
        """When requests finds nothing and fallback_to_playwright=True, retry with Playwright."""
        req_fetcher = MagicMock()
        req_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="")
        req_fetcher.fetch_images.return_value = []

        pw_fetcher = MagicMock()
        pw_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="playwright text")
        pw_fetcher.fetch_images.return_value = []

        mock_mf.side_effect = [req_fetcher, pw_fetcher]

        result = _web_to_text_command("http://example.com", selectors=[".sel"], fallback_to_playwright=True)
        assert result == "playwright text"
        assert mock_mf.call_count == 2
        mock_mf.assert_any_call(False, timeout=30, page_wait_for_timeout=0)
        mock_mf.assert_any_call(True, timeout=30, page_wait_for_timeout=0)

    @patch("wuzup.cli._make_fetcher")
    def test_fallback_not_triggered_when_requests_has_data(self, mock_mf):
        """Fallback is skipped when requests already found text."""
        req_fetcher = MagicMock()
        req_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="got it")
        req_fetcher.fetch_images.return_value = []
        mock_mf.return_value = req_fetcher

        result = _web_to_text_command("http://example.com", selectors=[".sel"], fallback_to_playwright=True)
        assert result == "got it"
        mock_mf.assert_called_once()  # No second call for Playwright

    @patch("wuzup.cli._make_fetcher")
    def test_fallback_both_empty_raises(self, mock_mf):
        """ValueError raised when both requests and Playwright fallback find nothing."""
        req_fetcher = MagicMock()
        req_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="")
        req_fetcher.fetch_images.return_value = []

        pw_fetcher = MagicMock()
        pw_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="")
        pw_fetcher.fetch_images.return_value = []

        mock_mf.side_effect = [req_fetcher, pw_fetcher]

        with pytest.raises(ValueError, match="No images or text found"):
            _web_to_text_command("http://example.com", selectors=[".sel"], fallback_to_playwright=True)

    @patch("wuzup.cli._make_fetcher")
    def test_fallback_not_used_when_already_playwright(self, mock_mf):
        """When use_playwright=True, fallback_to_playwright is a no-op (already using Playwright)."""
        pw_fetcher = MagicMock()
        pw_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="")
        pw_fetcher.fetch_images.return_value = []
        mock_mf.return_value = pw_fetcher

        with pytest.raises(ValueError, match="No images or text found"):
            _web_to_text_command(
                "http://example.com", selectors=[".sel"], use_playwright=True, fallback_to_playwright=True
            )
        mock_mf.assert_called_once()  # No retry

    def test_page_wait_without_playwright_or_fallback_raises(self):
        """page_wait_for_timeout without --playwright or -F raises ValueError."""
        with pytest.raises(
            ValueError, match="--page-wait-for-timeout requires --playwright or --fallback-to-playwright"
        ):
            _web_to_text_command("http://example.com", selectors=[".sel"], page_wait_for_timeout=5)

    @patch("wuzup.cli._make_fetcher")
    def test_page_wait_with_fallback_passes_to_playwright(self, mock_mf):
        """page_wait_for_timeout is forwarded to the Playwright fetcher during fallback."""
        req_fetcher = MagicMock()
        req_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="")
        req_fetcher.fetch_images.return_value = []

        pw_fetcher = MagicMock()
        pw_fetcher.scrape.return_value = ScrapeResult(img_urls=[], element_text="pw text")
        pw_fetcher.fetch_images.return_value = []

        mock_mf.side_effect = [req_fetcher, pw_fetcher]

        result = _web_to_text_command(
            "http://example.com", selectors=[".sel"], fallback_to_playwright=True, page_wait_for_timeout=3
        )
        assert result == "pw text"
        # requests fetcher gets page_wait_for_timeout=0
        mock_mf.assert_any_call(False, timeout=30, page_wait_for_timeout=0)
        # Playwright fetcher gets the actual value
        mock_mf.assert_any_call(True, timeout=30, page_wait_for_timeout=3)


# ===========================================================================
# cli.py — main (argument parsing)
# ===========================================================================


class TestMain:
    @patch("wuzup.cli._web_to_text_command", return_value="output")
    def test_web_to_text_alias_w2t(self, mock_cmd):
        buf = StringIO()
        main(["w2t", "-u", "http://example.com"], output=buf)
        mock_cmd.assert_called_once()
        assert buf.getvalue().strip() == "output"

    @patch("wuzup.cli._web_to_text_command", return_value="output")
    def test_web_to_text_alias_wtt(self, mock_cmd):
        buf = StringIO()
        main(["wtt", "-u", "http://example.com"], output=buf)
        mock_cmd.assert_called_once()

    @patch("wuzup.cli._web_to_text_command", return_value="output")
    def test_web_to_text_full_name(self, mock_cmd):
        buf = StringIO()
        main(["web-to-text", "-u", "http://example.com"], output=buf)
        mock_cmd.assert_called_once()

    @patch("wuzup.cli._web_to_text_command", return_value="text out")
    def test_passes_all_args(self, mock_cmd):
        buf = StringIO()
        main(
            [
                "web-to-text",
                "-u",
                "http://example.com",
                "-s",
                ".a",
                "-s",
                ".b",
                "-T",
                "15",
                "--page-wait-for-timeout",
                "2",
                "--playwright",
            ],
            output=buf,
        )
        mock_cmd.assert_called_once_with(
            url="http://example.com",
            selectors=[".a", ".b"],
            timeout=15.0,
            page_wait_for_timeout=2.0,
            use_playwright=True,
            fallback_to_playwright=False,
        )

    def test_missing_command_exits(self):
        with pytest.raises(SystemExit):
            main([])

    @patch("wuzup.cli._web_to_text_command", return_value="x")
    def test_fallback_flag_parsed(self, mock_cmd):
        buf = StringIO()
        main(["w2t", "-u", "http://example.com", "-s", ".a", "-F"], output=buf)
        mock_cmd.assert_called_once_with(
            url="http://example.com",
            selectors=[".a"],
            timeout=30.0,
            page_wait_for_timeout=0.0,
            use_playwright=False,
            fallback_to_playwright=True,
        )

    def test_playwright_and_fallback_mutually_exclusive(self):
        """--playwright and -F cannot be used together."""
        with pytest.raises(SystemExit):
            main(["w2t", "-u", "http://example.com", "--playwright", "-F"])

    def test_debug_flags(self):
        """--debug and --debug-all are parsed without errors."""
        with patch("wuzup.cli._web_to_text_command", return_value="x"):
            buf = StringIO()
            main(["--debug", "--debug-all", "w2t", "-u", "http://example.com"], output=buf)


# ===========================================================================
# CLI entry-point module
# ===========================================================================


class TestMainEntryPoint:
    def test_module_invocation(self):
        result = subprocess.run(
            [sys.executable, "-m", "wuzup", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "wuzup" in result.stdout


class TestVersion:
    def test_version_attribute(self):
        import wuzup

        assert hasattr(wuzup, "__version__")


# ===========================================================================
# Integration tests (marked so they can be skipped in CI without net)
# ===========================================================================


@pytest.mark.integration
class TestIntegrationRequestsFetcher:
    """Integration test using RequestsFetcher against a stable public URL."""

    def test_fetch_single_image(self):
        """Fetch a stable PNG from httpbin and verify it's a valid image."""
        from wuzup.requests_fetcher import RequestsFetcher

        fetcher = RequestsFetcher(timeout=30)
        img = fetcher.fetch_image("https://httpbin.org/image/png")
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_scrape_with_selectors(self):
        """Scrape example.com and verify text extraction from a real page."""
        from wuzup.requests_fetcher import RequestsFetcher

        fetcher = RequestsFetcher(timeout=30)
        result = fetcher.scrape("http://example.com", ["h1"])
        # example.com has a stable <h1> with text
        assert "Example Domain" in result.element_text


@pytest.mark.integration
class TestIntegrationPlaywrightFetcher:
    """Integration test using PlaywrightFetcher against a stable public URL."""

    def test_scrape_example_com(self):
        """Render example.com with Playwright and extract the heading."""
        from wuzup.playwright_fetcher import PlaywrightFetcher

        fetcher = PlaywrightFetcher(timeout=30)
        result = fetcher.scrape("http://example.com", ["h1"])
        assert "Example Domain" in result.element_text

    def test_fetch_image_via_playwright_fetcher(self):
        """PlaywrightFetcher.fetch_image (inherited) works with a real URL."""
        from wuzup.playwright_fetcher import PlaywrightFetcher

        fetcher = PlaywrightFetcher(timeout=30)
        img = fetcher.fetch_image("https://httpbin.org/image/png")
        assert isinstance(img, Image.Image)


@pytest.mark.integration
class TestIntegrationWebToText:
    """End-to-end integration test for the web-to-text CLI flow."""

    def test_direct_image_url(self):
        """web-to-text with a direct image URL (no selectors) produces text."""
        # httpbin.org/image/png is a stable image that shouldn't change.
        # We just confirm it runs without error and returns a string.
        result = _web_to_text_command("https://httpbin.org/image/png")
        assert isinstance(result, str)

    def test_selector_text_fallback(self):
        """web-to-text with example.com falls back to text since there are no matching images."""
        result = _web_to_text_command("http://example.com", selectors=["h1"])
        assert "Example Domain" in result
