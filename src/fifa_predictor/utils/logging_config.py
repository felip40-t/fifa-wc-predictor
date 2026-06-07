"""Centralized logging configuration for the fifa_predictor package."""

import logging

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically the caller's __name__.

    Returns:
        A logging.Logger instance configured with the package's handlers and level.
    """
    _configure()
    return logging.getLogger(name)
