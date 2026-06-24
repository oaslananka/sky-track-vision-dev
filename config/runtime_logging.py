from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PilotConfig

# ── ANSI color codes ─────────────────────────────────────────────
_COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41;37m",
    "TIMESTAMP": "\033[90m",
    "LOGGER": "\033[35m",
    "BRIDGE": "\033[34m",
    "TOOLS": "\033[33m",
    "PILOT": "\033[96m",
    "LLM": "\033[95m",
    "SAFETY": "\033[91m",
    "VISION": "\033[92m",
}

_COMPONENT_COLORS = {
    "skytrackvision.skypilot.bridge": _COLORS["BRIDGE"],
    "skytrackvision.skypilot.tools": _COLORS["TOOLS"],
    "skytrackvision.skypilot.pilot": _COLORS["PILOT"],
    "skytrackvision.skypilot.llm": _COLORS["LLM"],
    "skytrackvision.skypilot.display": _COLORS["VISION"],
    "skytrackvision.safety": _COLORS["SAFETY"],
    "skytrackvision.vision": _COLORS["VISION"],
}

_LEVEL_ICONS = {
    "DEBUG": "🔍",
    "INFO": "✅",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "🔥",
}

_CONSOLE_FIELD_ORDER = (
    "mission_id",
    "tick_id",
    "fsm_state",
    "tool_name",
    "request_id",
    "latency_ms",
    "target_id",
    "cmd_source",
    "safety_state",
    "altitude_m",
    "snapshot_age_ms",
    "reason",
    "applied",
)


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_jsonable(item) for item in value]
    return str(value)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "event_name": event,
            "event_fields": {key: _coerce_jsonable(value) for key, value in fields.items()},
        },
    )


def _extract_event_name(record: logging.LogRecord) -> str:
    return str(getattr(record, "event_name", "log"))


def _extract_event_fields(record: logging.LogRecord) -> dict[str, Any]:
    raw = getattr(record, "event_fields", {})
    if isinstance(raw, dict):
        return {str(key): _coerce_jsonable(value) for key, value in raw.items()}
    return {}


def _format_console_fields(fields: dict[str, Any]) -> str:
    ordered_pairs: list[str] = []
    for key in _CONSOLE_FIELD_ORDER:
        if key not in fields:
            continue
        ordered_pairs.append(f"{key}={fields[key]}")
    for key, value in fields.items():
        if key in _CONSOLE_FIELD_ORDER:
            continue
        ordered_pairs.append(f"{key}={value}")
    return " ".join(ordered_pairs)


class ColorFormatter(logging.Formatter):
    """Colorful console formatter with level-based coloring and component highlighting."""

    def format(self, record: logging.LogRecord) -> str:
        ts = _COLORS["TIMESTAMP"]
        lvl_color = _COLORS.get(record.levelname, _COLORS["RESET"])
        icon = _LEVEL_ICONS.get(record.levelname, "")
        rst = _COLORS["RESET"]

        comp_color = _COLORS["LOGGER"]
        for prefix, color in _COMPONENT_COLORS.items():
            if record.name.startswith(prefix):
                comp_color = color
                break

        short_name = record.name.replace("skytrackvision.", "")
        event_name = _extract_event_name(record)
        fields_text = _format_console_fields(_extract_event_fields(record))
        suffix = f" | {fields_text}" if fields_text else ""

        return (
            f"{ts}{self.formatTime(record, '%H:%M:%S')}{rst} "
            f"{lvl_color}{icon} {record.levelname:<8}{rst} "
            f"{comp_color}{short_name:<28}{rst} "
            f"{lvl_color}[{event_name}] {record.getMessage()}{suffix}{rst}"
        )


class FileFormatter(logging.Formatter):
    """Structured JSON-lines formatter for post-mission analysis."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "component": record.name,
            "event": _extract_event_name(record),
            "message": record.getMessage(),
            "func": record.funcName,
            "line": record.lineno,
        }
        payload.update(_extract_event_fields(record))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(cfg: PilotConfig) -> logging.Logger:
    """Configure colorful console + debug file logging."""
    level_name = cfg.log_level.upper()
    level = getattr(logging, level_name, logging.DEBUG)

    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # ── Console handler (user-facing, colored) ───────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColorFormatter())
    root.addHandler(console)

    # ── File handler (debug-level, for analysis) ─────────
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mission_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FileFormatter())
    root.addHandler(file_handler)

    logger = logging.getLogger("skytrackvision")
    logger.setLevel(logging.DEBUG)  # Always capture everything to file

    # Suppress noisy third-party
    for name in ("httpx", "httpcore", "openai", "openai._base_client", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.debug("Logging configured — console=%s, file=DEBUG → %s", level_name, log_file)
    logger.info("Log file: %s", log_file.resolve())
    return logger
