from __future__ import annotations

import logging

from config.runtime_logging import configure_logging
from config.settings import PilotConfig


def test_configure_logging_sets_debug_level_by_default() -> None:
    logger = configure_logging(PilotConfig())

    assert logger.level == logging.DEBUG


def test_configure_logging_respects_explicit_level() -> None:
    logger = configure_logging(PilotConfig(log_level="INFO"))

    # Logger is always DEBUG (for file logging), console handler has the explicit level
    assert logger.level == logging.DEBUG
    root = logging.getLogger()
    console_handlers = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]
    assert any(h.level == logging.INFO for h in console_handlers)
