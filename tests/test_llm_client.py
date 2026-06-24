from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config.settings import PilotConfig
from skypilot.llm_client import OpenAIProviderAdapter, _supports_sampling_overrides, resolve_api_key


def test_resolve_api_key_reads_local_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")

    api_key = resolve_api_key(PilotConfig(provider="openai"))

    assert api_key == "test-key"


def test_resolve_api_key_raises_clear_error_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_api_key(PilotConfig(provider="openai"))


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4o", True),
        ("gpt-4o-mini", True),
        ("gpt-3.5-turbo", True),
        ("gpt-5-mini", False),
        ("gpt-5", False),
        ("o1", False),
        ("o1-mini", False),
        ("o3", False),
        ("o3-mini", False),
        ("O1-MINI", False),  # case-insensitive
        ("GPT-5-MINI", False),  # case-insensitive
    ],
)
def test_supports_sampling_overrides_parametrized(model: str, expected: bool) -> None:
    assert _supports_sampling_overrides(model) is expected


async def test_chat_raises_value_error_on_null_message() -> None:
    """If choices[0].message is None, chat() must raise ValueError."""
    from openai import AsyncOpenAI

    mock_client = MagicMock(spec=AsyncOpenAI)
    adapter = OpenAIProviderAdapter(
        client=mock_client,
        model="gpt-4o",
        temperature=1.0,
        top_p=1.0,
        max_tokens=256,
    )

    # Build a mock response with message=None.
    mock_choice = MagicMock()
    mock_choice.message = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def _fake_create(**_kwargs):
        return mock_response

    mock_client.chat.completions.create = _fake_create

    with pytest.raises(ValueError, match="choices\\[0\\].message is None"):
        await adapter.chat(messages=[], tools=[], system="")
