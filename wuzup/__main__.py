import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from wuzup.ai import ask_ai_for_boolean_response
from wuzup.cache import Cache
from wuzup.search import Searcher

log = logging.getLogger(__name__)

# By default we'll have info warning logs globally
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# For wuzup specific logs, default to INFO
wuzup_logger = logging.getLogger("wuzup")
wuzup_logger.setLevel(logging.INFO)


def _ask_loop():
    """For debug."""
    while True:
        q = input("Ask a yes/no question: ")
        print(ask_ai_for_boolean_response(q))


def _debug_setup(debug: bool, debug_all: bool):
    """
    Mockable function for messing with enabling debug logging.
    debug: if true, enable wuzup debug logging
    debug_all: if true, enable all debug logging globally
    """
    log.setLevel(logging.DEBUG)

    if debug:
        log.info("Turning on debug logging via --debug")
        logging.getLogger("wuzup").setLevel(logging.DEBUG)
    if debug_all:
        log.info("Turning on ALL debug logging via --debug-all")
        logging.getLogger().setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(description="wuzup cli")
    parser.add_argument("--debug", action="store_true", help="If given, turn on wuzup debug logging.")
    parser.add_argument("--debug-all", action="store_true", help="If given, turn on ALL debug logging globally.")
    parser.add_argument("--skip-search", action="store_true", help="If given, skip the search step.")
    parser.add_argument("--cache", "-c", type=Path, help="If given, specify the cache directory.")
    parser.add_argument("--query", "-q", type=str, required=True, help="The search query to execute.")
    args = parser.parse_args()

    _debug_setup(args.debug, args.debug_all)
    # _ask_loop()

    if args.cache:
        cache = Cache(args.cache)
        log.info(f"Using cache directory: {args.cache}")
    else:
        with TemporaryDirectory(prefix="wuzup_cache_") as temp_cache_dir:
            cache = Cache(Path(temp_cache_dir))
            log.info(f"Using temporary cache directory: {temp_cache_dir}")

    if not args.skip_search:
        added_count = 0
        searcher = Searcher(args.query, typ="news", search_kwargs={"timelimit": "w"})
        results = searcher.search()

        for result in results:
            added = cache.add_to_topic(args.query, result, safe=True)
            if added:
                log.info(f"Added result to cache: {result.title}")
                added_count += 1
            else:
                log.info(f"Result already in cache, skipped: {result.title}")

        log.info(f"Added {added_count} new results to cache.")

    log.info(f"Dumping cache for topic '{args.query}': ")
    for item in cache.dump_topic(args.query):
        log.info(f"- {item.data.title} ({item.data.url}) - {item.data}")


if __name__ == "__main__":
    main()
