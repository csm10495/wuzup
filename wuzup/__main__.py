import argparse
import logging

from wuzup.ai import ask_ai_for_boolean_response

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
    args = parser.parse_args()

    _debug_setup(args.debug, args.debug_all)
    _ask_loop()


if __name__ == "__main__":
    main()
