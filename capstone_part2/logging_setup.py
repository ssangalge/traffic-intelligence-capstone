"""
logging_setup.py

Standalone logging configuration for the capstone Part 2 pipeline.
Run this file directly to verify the logging setup works correctly
before it gets used inside pipeline.py.

Requirements this satisfies:
- Module-level logger (logging.getLogger(__name__))
- INFO / WARNING / ERROR / DEBUG all demonstrated
- Logs to BOTH console and a file (pipeline.log)
- Formatter includes timestamp, level, module name, and message
- No print() used for status/log-like output
"""

import logging
import sys


def configure_logging(log_file: str = "pipeline.log", level=logging.DEBUG) -> logging.Logger:
    """Configure root logging behaviour: console + file, with a consistent format.

    Call this ONCE, early in your program's entry point (e.g. at the top of
    pipeline.py's __main__ block). Every other module should just call
    logging.getLogger(__name__) and use it -- they inherit this configuration
    automatically without needing to reconfigure anything themselves.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if this is called more than once
    # (e.g. during interactive testing)
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Console handler -- shows INFO and above by default, keeps terminal readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler -- captures everything including DEBUG, for later review
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Third-party libraries (e.g. matplotlib) emit very verbose DEBUG logs
    # (font-matching internals, etc.) that add noise without adding value.
    # Raise their threshold so only WARNING+ from them appears, while our
    # own application code still logs at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root_logger


# Module-level logger for THIS file, used the same way every other module will use it
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    configure_logging()

    logger.debug("This is a DEBUG message -- detailed diagnostic info, file only by default.")
    logger.info("This is an INFO message -- normal progress update, shown in console and file.")
    logger.warning("This is a WARNING message -- something unexpected but not fatal.")
    logger.error("This is an ERROR message -- something failed.")

    logger.info("Logging setup verified. Check pipeline.log to confirm DEBUG lines were captured there too.")