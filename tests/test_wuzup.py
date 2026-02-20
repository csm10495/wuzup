"""Comprehensive unit tests for wuzup."""

import logging
import subprocess
import sys
from io import BytesIO, StringIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from wuzup.cli import (
    _debug_setup,
    _images_to_text_command,
    main,
)
from wuzup.image import (
    composite_on_color,
    image_to_text,
    load_image_from_path,
    load_images_from_url,
    ocr_variant,
)

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


# ---------------------------------------------------------------------------
# _debug_setup
# ---------------------------------------------------------------------------


class TestDebugSetup:
    def test_no_flags(self):
        """No flags should leave log levels unchanged."""
        with patch.object(logging.getLogger("wuzup"), "setLevel") as mock_set:
            _debug_setup(False, False)
            mock_set.assert_not_called()

    def test_debug_flag(self):
        """--debug should set wuzup logger to DEBUG."""
        _debug_setup(True, False)
        assert logging.getLogger("wuzup").level == logging.DEBUG
        # Reset
        logging.getLogger("wuzup").setLevel(logging.INFO)

    def test_debug_all_flag(self):
        """--debug-all should set the root logger to DEBUG."""
        original = logging.getLogger().level
        _debug_setup(False, True)
        assert logging.getLogger().level == logging.DEBUG
        # Reset
        logging.getLogger().setLevel(original)

    def test_both_flags(self):
        """Both flags together should set both loggers."""
        original_root = logging.getLogger().level
        _debug_setup(True, True)
        assert logging.getLogger("wuzup").level == logging.DEBUG
        assert logging.getLogger().level == logging.DEBUG
        # Reset
        logging.getLogger("wuzup").setLevel(logging.INFO)
        logging.getLogger().setLevel(original_root)


# ---------------------------------------------------------------------------
# _load_image_from_path
# ---------------------------------------------------------------------------


class TestLoadImageFromPath:
    def test_loads_image(self, tmp_path):
        """Should open a local image file and return a PIL Image."""
        img = _make_rgb_image()
        path = tmp_path / "test.png"
        img.save(path)
        result = load_image_from_path(str(path))
        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image_from_path("/nonexistent/path.png")


# ---------------------------------------------------------------------------
# _load_images_from_url
# ---------------------------------------------------------------------------


class TestLoadImagesFromUrl:
    def test_direct_url(self):
        """Without selectors, should fetch the URL directly as an image."""
        img_bytes = _image_to_bytes(_make_rgb_image())
        mock_resp = MagicMock()
        mock_resp.content = img_bytes
        mock_resp.raise_for_status = MagicMock()

        def fake_get(url, **kwargs):
            assert url == "http://example.com/img.png"
            assert kwargs == {"timeout": 30}
            return mock_resp

        with patch("wuzup.image.requests.get", side_effect=fake_get):
            result = load_images_from_url("http://example.com/img.png")
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)

    def test_with_selector_finds_src(self):
        """With a selector, should parse HTML page and follow src attribute."""
        html = '<html><body><img id="target" src="/images/photo.png"></body></html>'
        img_bytes = _image_to_bytes(_make_rgb_image())

        page_resp = MagicMock()
        page_resp.text = html
        page_resp.raise_for_status = MagicMock()

        img_resp = MagicMock()
        img_resp.content = img_bytes
        img_resp.raise_for_status = MagicMock()

        responses = iter([page_resp, img_resp])

        with patch("wuzup.image.requests.get", side_effect=lambda url, **kw: next(responses)) as mock_get:
            result = load_images_from_url("http://example.com/page", selectors=["#target"])
            assert mock_get.call_count == 2
            # Second call should resolve relative URL
            mock_get.assert_called_with("http://example.com/images/photo.png", timeout=30)
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)

    def test_with_selector_finds_data_src(self):
        """Should fall back to data-src when src is absent."""
        html = '<html><body><img class="lazy" data-src="https://cdn.example.com/pic.jpg"></body></html>'
        img_bytes = _image_to_bytes(_make_rgb_image())

        page_resp = MagicMock()
        page_resp.text = html
        page_resp.raise_for_status = MagicMock()

        img_resp = MagicMock()
        img_resp.content = img_bytes
        img_resp.raise_for_status = MagicMock()

        responses = iter([page_resp, img_resp])

        with patch("wuzup.image.requests.get", side_effect=lambda url, **kw: next(responses)):
            result = load_images_from_url("http://example.com/page", selectors=[".lazy"])
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)

    def test_with_selector_no_match_raises(self):
        """Should raise ValueError when selector matches no element."""
        html = "<html><body><p>No images here</p></body></html>"
        page_resp = MagicMock()
        page_resp.text = html
        page_resp.raise_for_status = MagicMock()

        with patch("wuzup.image.requests.get", return_value=page_resp):
            with pytest.raises(ValueError, match="No images found matching any selector"):
                load_images_from_url("http://example.com/page", selectors=["#missing"])

    def test_with_selector_element_no_src_skipped(self):
        """Element with no src/data-src should be skipped; raises if no images found."""
        html = '<html><body><div id="nosrc">hello</div></body></html>'
        page_resp = MagicMock()
        page_resp.text = html
        page_resp.raise_for_status = MagicMock()

        with patch("wuzup.image.requests.get", return_value=page_resp):
            with pytest.raises(ValueError, match="No images found matching any selector"):
                load_images_from_url("http://example.com/page", selectors=["#nosrc"])

    def test_relative_url_joined(self):
        """Relative src should be resolved against the base page URL."""
        html = '<html><body><img id="img" src="relative/pic.png"></body></html>'
        img_bytes = _image_to_bytes(_make_rgb_image())

        page_resp = MagicMock()
        page_resp.text = html
        page_resp.raise_for_status = MagicMock()

        img_resp = MagicMock()
        img_resp.content = img_bytes
        img_resp.raise_for_status = MagicMock()

        responses = iter([page_resp, img_resp])
        calls = []

        def fake_get(url, **kw):
            calls.append(url)
            return next(responses)

        with patch("wuzup.image.requests.get", side_effect=fake_get):
            load_images_from_url("http://example.com/dir/page", selectors=["#img"])
            assert calls[-1] == "http://example.com/dir/relative/pic.png"

    def test_http_error_propagated(self):
        """requests exceptions should propagate."""
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

        with patch("wuzup.image.requests.get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                load_images_from_url("http://example.com/bad.png")


# ---------------------------------------------------------------------------
# _composite_on_color
# ---------------------------------------------------------------------------


class TestCompositeOnColor:
    def test_rgba_composited(self):
        """RGBA image should be composited onto the given color."""
        img = _make_rgba_image(size=(4, 4), color=(0, 0, 0, 0))  # fully transparent
        result = composite_on_color(img, (255, 0, 0))
        assert result.mode == "RGB"
        # Fully transparent over red background -> red
        assert result.getpixel((0, 0)) == (255, 0, 0)

    def test_rgb_converted_to_rgba(self):
        """RGB input should be converted to RGBA first."""
        img = _make_rgb_image(size=(4, 4), color=(0, 255, 0))
        result = composite_on_color(img, (0, 0, 255))
        assert result.mode == "RGB"
        # Opaque green composited on blue -> green
        assert result.getpixel((0, 0)) == (0, 255, 0)

    def test_opaque_pixel_preserved(self):
        """Fully opaque pixels should show through unchanged."""
        img = _make_rgba_image(size=(2, 2), color=(42, 100, 200, 255))
        result = composite_on_color(img, (0, 0, 0))
        assert result.getpixel((0, 0)) == (42, 100, 200)

    def test_output_size_matches_input(self):
        img = _make_rgba_image(size=(7, 13))
        result = composite_on_color(img, (128, 128, 128))
        assert result.size == (7, 13)


# ---------------------------------------------------------------------------
# _ocr_variant
# ---------------------------------------------------------------------------


class TestOcrVariant:
    def test_calls_pytesseract(self):
        """Should delegate to pytesseract and strip the result."""
        img = _make_rgb_image()
        with patch("wuzup.image.pytesseract.image_to_string", return_value="  Hello World  \n") as mock_ocr:
            result = ocr_variant(img)
            mock_ocr.assert_called_once_with(img)
            assert result == "Hello World"


# ---------------------------------------------------------------------------
# _image_to_text
# ---------------------------------------------------------------------------


class TestImageToText:
    def test_deduplicates_lines(self):
        """Lines appearing in multiple variants should appear only once."""
        img = _make_rgba_image()

        with patch("wuzup.image.ocr_variant", return_value="Hello\nWorld"):
            result = image_to_text(img)
            assert result == "Hello\nWorld"

    def test_deduplication_is_case_insensitive(self):
        """Dedup key uses lowercase, but original casing is preserved."""
        img = _make_rgba_image()

        side_effects = [
            "HELLO",
            "hello",
            "Hello",
            "",
            "",
        ]

        with patch("wuzup.image.ocr_variant", side_effect=side_effects):
            result = image_to_text(img)
            assert result == "HELLO"  # first occurrence wins

    def test_empty_lines_skipped(self):
        """Blank lines from OCR should be filtered out."""
        img = _make_rgba_image()

        with patch("wuzup.image.ocr_variant", return_value="\n\n   \n"):
            result = image_to_text(img)
            assert result == ""

    def test_five_variants_processed(self):
        """Should create 5 variants: white bg, black bg, R, G, B channels."""
        img = _make_rgba_image()

        with patch("wuzup.image.ocr_variant", return_value="") as mock_ocr:
            image_to_text(img)
            assert mock_ocr.call_count == 5

    def test_collects_unique_lines_across_variants(self):
        """Lines from different variants should be merged."""
        img = _make_rgba_image()

        side_effects = [
            "line from white",
            "line from black",
            "line from R",
            "line from G",
            "line from B",
        ]

        with patch("wuzup.image.ocr_variant", side_effect=side_effects):
            result = image_to_text(img)
            lines = result.splitlines()
            assert lines == [
                "line from white",
                "line from black",
                "line from R",
                "line from G",
                "line from B",
            ]


# ---------------------------------------------------------------------------
# _images_to_text_command
# ---------------------------------------------------------------------------


class TestImageToTextCommand:
    def test_path_mode(self):
        """When path is given, should load from path and return OCR result."""
        mock_img = _make_rgb_image()
        with (
            patch("wuzup.cli.load_image_from_path", return_value=mock_img) as mock_load,
            patch("wuzup.cli.image_to_text", return_value="OCR result") as mock_ocr,
        ):
            result = _images_to_text_command(path="/some/img.png")
            mock_load.assert_called_once_with("/some/img.png")
            mock_ocr.assert_called_once_with(mock_img)
        assert result == "OCR result"

    def test_url_mode(self):
        """When url is given, should load from URL and return OCR result."""
        mock_img = _make_rgb_image()
        with (
            patch("wuzup.cli.load_images_from_url", return_value=[mock_img]) as mock_load,
            patch("wuzup.cli.image_to_text", return_value="URL text") as mock_ocr,
        ):
            result = _images_to_text_command(url="http://example.com/img.png")
            mock_load.assert_called_once_with("http://example.com/img.png", None)
            mock_ocr.assert_called_once_with(mock_img)
        assert result == "URL text"

    def test_url_with_selector(self):
        """Selectors should be passed through to load_images_from_url."""
        mock_img = _make_rgb_image()
        with (
            patch("wuzup.cli.load_images_from_url", return_value=[mock_img]) as mock_load,
            patch("wuzup.cli.image_to_text", return_value="selected"),
        ):
            result = _images_to_text_command(url="http://example.com/page", selectors=["#main-img"])
            mock_load.assert_called_once_with("http://example.com/page", ["#main-img"])
        assert result == "selected"


# ---------------------------------------------------------------------------
# main() / CLI argument parsing
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_exits(self):
        """No arguments should cause SystemExit (required subcommand)."""
        with patch("sys.argv", ["wuzup"]):
            with pytest.raises(SystemExit):
                main()

    def test_image_to_text_path(self):
        """image-to-text --path should call _images_to_text_command."""
        with (
            patch("sys.argv", ["wuzup", "image-to-text", "--path", "/tmp/img.png"]),
            patch("wuzup.cli._images_to_text_command", return_value="") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once_with(path="/tmp/img.png", url=None, selectors=None)

    def test_i2t_alias(self):
        """i2t alias should work identically."""
        with (
            patch("sys.argv", ["wuzup", "i2t", "--path", "/tmp/img.png"]),
            patch("wuzup.cli._images_to_text_command", return_value="") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once()

    def test_itt_alias(self):
        """itt alias should work identically."""
        with (
            patch("sys.argv", ["wuzup", "itt", "--url", "http://example.com/img.png"]),
            patch("wuzup.cli._images_to_text_command", return_value="") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once_with(path=None, url="http://example.com/img.png", selectors=None)

    def test_url_with_selector(self):
        with (
            patch("sys.argv", ["wuzup", "image-to-text", "--url", "http://example.com", "-s", "img.hero"]),
            patch("wuzup.cli._images_to_text_command", return_value="") as mock_cmd,
        ):
            main()
            mock_cmd.assert_called_once_with(path=None, url="http://example.com", selectors=["img.hero"])

    def test_main_prints_command_result(self, capsys):
        """main() should print the return value of _images_to_text_command."""
        with (
            patch("sys.argv", ["wuzup", "image-to-text", "--path", "/tmp/img.png"]),
            patch("wuzup.cli._images_to_text_command", return_value="printed text"),
        ):
            main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "printed text"

    def test_selector_without_url_exits(self):
        """--selector with --path (no --url) should cause SystemExit."""
        with patch("sys.argv", ["wuzup", "image-to-text", "--path", "/tmp/x.png", "-s", "img"]):
            with pytest.raises(SystemExit):
                main()

    def test_path_and_url_mutually_exclusive(self):
        """--path and --url together should cause SystemExit."""
        with patch("sys.argv", ["wuzup", "image-to-text", "--path", "x", "--url", "y"]):
            with pytest.raises(SystemExit):
                main()

    def test_neither_path_nor_url_exits(self):
        """Omitting both --path and --url should cause SystemExit."""
        with patch("sys.argv", ["wuzup", "image-to-text"]):
            with pytest.raises(SystemExit):
                main()

    def test_debug_flag_passed(self):
        with (
            patch("sys.argv", ["wuzup", "--debug", "image-to-text", "--path", "/tmp/x.png"]),
            patch("wuzup.cli._images_to_text_command", return_value=""),
            patch("wuzup.cli._debug_setup") as mock_debug,
        ):
            main()
            mock_debug.assert_called_once_with(True, False)

    def test_debug_all_flag_passed(self):
        with (
            patch("sys.argv", ["wuzup", "--debug-all", "image-to-text", "--path", "/tmp/x.png"]),
            patch("wuzup.cli._images_to_text_command", return_value=""),
            patch("wuzup.cli._debug_setup") as mock_debug,
        ):
            main()
            mock_debug.assert_called_once_with(False, True)

    def test_both_debug_flags(self):
        with (
            patch("sys.argv", ["wuzup", "--debug", "--debug-all", "image-to-text", "--path", "/tmp/x.png"]),
            patch("wuzup.cli._images_to_text_command", return_value=""),
            patch("wuzup.cli._debug_setup") as mock_debug,
        ):
            main()
            mock_debug.assert_called_once_with(True, True)

    def test_explicit_args_parameter(self, capsys):
        """Passing args directly should bypass sys.argv."""
        with patch("wuzup.cli._images_to_text_command", return_value="from args") as mock_cmd:
            main(["image-to-text", "--path", "/tmp/img.png"])
            mock_cmd.assert_called_once_with(path="/tmp/img.png", url=None, selectors=None)
        assert capsys.readouterr().out.strip() == "from args"

    def test_output_parameter(self):
        """Passing output should write there instead of stdout."""
        buf = StringIO()
        with patch("wuzup.cli._images_to_text_command", return_value="to buffer"):
            main(["image-to-text", "--path", "/tmp/img.png"], output=buf)
        assert buf.getvalue().strip() == "to buffer"


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_run_as_module(self):
        """python -m wuzup with no args should exit with error."""
        result = subprocess.run(
            [sys.executable, "-m", "wuzup"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_run_as_module_help(self):
        """python -m wuzup --help should show help and exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "wuzup", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "wuzup" in result.stdout.lower()

    def test_run_cli_directly(self):
        """Running cli.py directly should work as an entry point."""
        result = subprocess.run(
            [sys.executable, "-c", "from wuzup.cli import main; main()", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "wuzup" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_defined(self):
        from wuzup import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0


# ---------------------------------------------------------------------------
# Integration tests (require network + tesseract)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_load_images_from_real_url(self):
        """Fetch a real image from the web."""
        url = "https://httpbin.org/image/png"
        images = load_images_from_url(url)
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
