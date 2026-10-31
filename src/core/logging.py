"""Logging setup shared by the server and the command-line scripts."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, datefmt="%H:%M:%S")
    # These libraries log every HTTP request at INFO, which drowns out our own lines
    for noisy in ("httpx", "pymongo", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
