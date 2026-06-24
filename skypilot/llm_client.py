from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    from openai import APIError, AsyncOpenAI, RateLimitError
except ModuleNotFoundError:  # pragma: no cover - optional dependency for local testing/docs

    class APIError(Exception):  # type: ignore[no-redef]
        """Fallback API error used when the OpenAI SDK is unavailable."""

    class RateLimitError(Exception):  # type: ignore[no-redef]
        """Fallback rate-limit error used when the OpenAI SDK is unavailable."""

    AsyncOpenAI = Any  # type: ignore[assignment,misc]

from config.runtime_logging import log_event
from config.settings import PilotConfig
from skypilot.models import ChatResponse, ToolCall

logger = logging.getLogger("skytrackvision.skypilot.llm")


@dataclass(slots=True)
class OpenAIProviderAdapter:
    client: AsyncOpenAI
    model: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> ChatResponse:
        payload = [{"role": "system", "content": system}, *messages]
        request_id = f"llm-{time.monotonic_ns()}"
        prompt_chars = sum(len(str(message.get("content", ""))) for message in payload)
        user_preview = _extract_last_user_preview(payload)
        log_event(
            logger,
            logging.DEBUG,
            "llm.request",
            "Sending LLM request",
            request_id=request_id,
            model=self.model,
            message_count=len(payload),
            tool_count=len(tools),
            prompt_chars=prompt_chars,
            user_preview=user_preview,
        )

        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                started = time.monotonic()
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": payload,
                    "tools": tools or None,
                    "tool_choice": "required" if tools else None,
                    "parallel_tool_calls": False if tools else None,
                    # OpenAI API v2025: use max_completion_tokens.
                    "max_completion_tokens": self.max_tokens,
                }

                # gpt-5-* chat.completions can reject non-default sampling overrides.
                if _supports_sampling_overrides(self.model):
                    request_kwargs["temperature"] = self.temperature
                    request_kwargs["top_p"] = self.top_p

                response = await self.client.chat.completions.create(
                    **request_kwargs,
                )

                # Validate response structure
                if not response.choices:
                    raise ValueError("Empty response from LLM - no choices returned")

                message = response.choices[0].message
                if message is None:
                    raise ValueError(
                        "LLM response choices[0].message is None — malformed API response"
                    )
                tool_names = [call.function.name for call in (message.tool_calls or [])]
                log_event(
                    logger,
                    logging.DEBUG,
                    "llm.response",
                    "Received LLM response",
                    request_id=request_id,
                    latency_ms=round((time.monotonic() - started) * 1000, 2),
                    content_chars=len(message.content or ""),
                    tool_calls=len(message.tool_calls or []),
                    tool_names=tool_names,
                    assistant_preview=_compact_preview(message.content),
                    finish_reason=response.choices[0].finish_reason,
                )

                tool_calls: list[ToolCall] = []
                if message.tool_calls:
                    for call in message.tool_calls:
                        try:
                            arguments = json.loads(call.function.arguments or "{}")
                        except json.JSONDecodeError as e:
                            logger.error(
                                "Failed to parse tool call arguments for '%s' — "
                                "continuing with empty args: %s",
                                call.function.name,
                                e,
                            )
                            arguments = {}

                        tool_calls.append(
                            cast(
                                ToolCall,
                                {
                                    "id": call.id,
                                    "name": call.function.name,
                                    "arguments": arguments,
                                },
                            )
                        )

                return {
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                }

            except (APIError, RateLimitError) as e:
                if attempt == max_retries - 1:
                    log_event(
                        logger,
                        logging.ERROR,
                        "llm.response",
                        "LLM API error after retries",
                        request_id=request_id,
                        attempts=max_retries,
                        reason=str(e),
                    )
                    raise
                wait_time = (2**attempt) * random.uniform(0.75, 1.25)
                log_event(
                    logger,
                    logging.WARNING,
                    "llm.retry",
                    "LLM API error, retrying",
                    request_id=request_id,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time_s=wait_time,
                    reason=str(e),
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                log_event(
                    logger,
                    logging.ERROR,
                    "llm.response",
                    "Unexpected LLM error",
                    request_id=request_id,
                    reason=str(e),
                )
                raise
        raise RuntimeError("LLM request loop exited unexpectedly")


def build_llm_client(cfg: PilotConfig) -> OpenAIProviderAdapter:
    if AsyncOpenAI is Any:
        raise RuntimeError(
            "The 'openai' package is required for SkyPilot mission execution. "
            "Install the project runtime dependencies first."
        )
    api_key = resolve_api_key(cfg)
    log_event(
        logger,
        logging.DEBUG,
        "llm.client",
        "Building LLM client",
        provider=cfg.provider,
        model=cfg.model,
        temperature=cfg.llm_temperature,
        top_p=cfg.llm_top_p,
        max_tokens=cfg.llm_max_tokens,
    )
    if cfg.provider == "github":
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://models.inference.ai.azure.com",
        )
        return OpenAIProviderAdapter(
            client=client,
            model=cfg.model,
            temperature=cfg.llm_temperature,
            top_p=cfg.llm_top_p,
            max_tokens=cfg.llm_max_tokens,
        )
    client = AsyncOpenAI(api_key=api_key)
    return OpenAIProviderAdapter(
        client=client,
        model=cfg.model,
        temperature=cfg.llm_temperature,
        top_p=cfg.llm_top_p,
        max_tokens=cfg.llm_max_tokens,
    )


def resolve_api_key(cfg: PilotConfig) -> str:
    """Resolve provider credentials from the environment, then from a local .env file."""
    env_name = "GITHUB_TOKEN" if cfg.provider == "github" else "OPENAI_API_KEY"
    direct_value = os.getenv(env_name)
    if direct_value:
        log_event(
            logger,
            logging.DEBUG,
            "llm.auth",
            "Resolved API key from environment",
            env=env_name,
        )
        return direct_value
    dotenv_value = _read_dotenv_value(Path.cwd() / ".env", env_name)
    if dotenv_value:
        os.environ.setdefault(env_name, dotenv_value)
        log_event(
            logger,
            logging.DEBUG,
            "llm.auth",
            "Resolved API key from local .env",
            env=env_name,
        )
        return dotenv_value
    raise RuntimeError(
        f"{env_name} is required for provider='{cfg.provider}'. "
        "Set it in the environment or add it to a local .env file."
    )


def _read_dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        cleaned = value.strip().strip('"').strip("'")
        return cleaned or None
    return None


def _extract_last_user_preview(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")) != "user":
            continue
        return _compact_preview(message.get("content"))
    return ""


def _compact_preview(content: Any, *, max_chars: int = 96) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        text = " ".join(part for part in parts if part)
    else:
        text = str(content)

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _supports_sampling_overrides(model: str) -> bool:
    """Return True when explicit temperature/top_p should be sent.

    Reasoning-series models (o1, o3, o1-mini, o3-mini) and gpt-5 variants
    reject temperature/top_p overrides — omit them so the API uses its defaults.
    """
    normalized = model.strip().lower()
    _no_sampling = ("o1", "o3", "o1-mini", "o3-mini", "gpt-5")
    return not any(normalized.startswith(p) for p in _no_sampling)
