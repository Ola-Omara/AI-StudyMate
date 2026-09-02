import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    root_logger = logging.getLogger("ai_studymate")
    root_logger.setLevel(log_level.upper())

    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ai_studymate.{name}")
