"""Command-line interface for wuzup."""

import argparse
import logging

from wuzup.image import image_to_text, load_image_from_path, load_image_from_url

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
        logging.getLogger("wuzup").setLevel(logging.DEBUG)
    if debug_all:
        log.info("Turning on ALL debug logging via --debug-all")
        logging.getLogger().setLevel(logging.DEBUG)


def _image_to_text_command(
    path: str | None = None,
    url: str | None = None,
    selector: str | None = None,
) -> str:
    """Load an image and extract text from it via OCR.

    Exactly one of *path* or *url* must be provided.

    Args:
        path: Filesystem path to a local image file.
        url: URL pointing to an image or an HTML page containing one.
        selector: Optional CSS selector to locate an ``<img>`` element
            on the page (only used with *url*).

    Returns:
        The OCR-extracted text.
    """
    if path:
        image = load_image_from_path(path)
    else:
        image = load_image_from_url(url, selector)

    return image_to_text(image)


def main():
    """Entry point for the ``wuzup`` CLI.

    Parses command-line arguments, configures logging, and dispatches to
    the appropriate sub-command handler.
    """
    parser = argparse.ArgumentParser(description="wuzup - a multi-tool CLI")
    parser.add_argument("--debug", action="store_true", help="If given, turn on wuzup debug logging.")
    parser.add_argument("--debug-all", action="store_true", help="If given, turn on ALL debug logging globally.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    itt_parser = subparsers.add_parser(
        "image-to-text", aliases=["i2t", "itt"], help="Extract text from an image using OCR."
    )
    source_group = itt_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-p", "--path", type=str, help="Path to a local image file.")
    source_group.add_argument("-u", "--url", type=str, help="URL to an image or a page containing an image.")
    itt_parser.add_argument(
        "-s", "--selector", type=str, help="CSS selector to find the image on the page (only with --url)."
    )

    args = parser.parse_args()
    _debug_setup(args.debug, args.debug_all)

    if args.command in ("image-to-text", "i2t", "itt"):
        if args.selector and not args.url:
            parser.error("--selector can only be used with --url")
        print(_image_to_text_command(path=args.path, url=args.url, selector=args.selector))


if __name__ == "__main__":
    main()
