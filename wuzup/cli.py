"""Command-line interface for wuzup."""

import argparse
import io
import logging
import sys

from wuzup.image import image_to_text, load_image_from_path, load_images_from_url

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


def _images_to_text_command(
    path: str | None = None,
    url: str | None = None,
    selectors: list[str] | None = None,
) -> str:
    """Load image(s) and extract text via OCR.

    Exactly one of *path* or *url* must be provided.

    Args:
        path: Filesystem path to a local image file.
        url: URL pointing to an image or an HTML page containing one.
        selectors: Optional list of CSS selectors to locate ``<img>``
            elements on the page (only used with *url*).  Images matching
            *any* selector are processed.

    Returns:
        The OCR-extracted text from all matched images.
    """
    if path:
        images = [load_image_from_path(path)]
    else:
        images = load_images_from_url(url, selectors)

    all_lines: list[str] = []
    seen: set[str] = set()
    for image in images:
        text = image_to_text(image)
        for line in text.splitlines():
            key = line.lower()
            if key not in seen:
                seen.add(key)
                all_lines.append(line)

    return "\n".join(all_lines)


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

    itt_parser = subparsers.add_parser(
        "image-to-text", aliases=["i2t", "itt"], help="Extract text from an image using OCR."
    )
    source_group = itt_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-p", "--path", type=str, help="Path to a local image file.")
    source_group.add_argument("-u", "--url", type=str, help="URL to an image or a page containing an image.")
    itt_parser.add_argument(
        "-s",
        "--selector",
        type=str,
        action="append",
        dest="selectors",
        help="CSS selector to find an image on the page (only with --url). Can be specified multiple times.",
    )

    args = parser.parse_args(args)
    _debug_setup(args.debug, args.debug_all)

    if args.command in ("image-to-text", "i2t", "itt"):
        if args.selectors and not args.url:
            parser.error("--selector can only be used with --url")
        print(_images_to_text_command(path=args.path, url=args.url, selectors=args.selectors), file=output)


if __name__ == "__main__":
    main()
