from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from airsim_control import client as client_module
from airsim_control.client import AirSimConnectionManager
from config.settings import AirSimConfig


class _SuccessfulClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.api_enabled = False
        self.armed = False

    def confirmConnection(self) -> None:  # noqa: N802
        return None

    def enableApiControl(self, enabled: bool, vehicle_name: str | None = None) -> None:  # noqa: N802
        self.api_enabled = enabled

    def armDisarm(self, armed: bool, vehicle_name: str | None = None) -> None:  # noqa: N802
        self.armed = armed


class _HangingClient(_SuccessfulClient):
    def confirmConnection(self) -> None:  # noqa: N802
        while True:
            time.sleep(0.05)


def test_connect_succeeds_with_timeout_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_airsim = type("FakeAirSim", (), {"MultirotorClient": _SuccessfulClient})
    monkeypatch.setattr(client_module, "airsim", fake_airsim)
    manager = AirSimConnectionManager(AirSimConfig(timeout_s=0.2))

    client = manager.connect(retries=1)

    assert client is manager.client
    assert manager.state.connected is True
    assert client.api_enabled is True
    assert client.armed is True


def test_connect_times_out_instead_of_blocking_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_airsim = type("FakeAirSim", (), {"MultirotorClient": _HangingClient})
    monkeypatch.setattr(client_module, "airsim", fake_airsim)
    manager = AirSimConnectionManager(AirSimConfig(timeout_s=0.1))

    started = time.monotonic()
    with pytest.raises(ConnectionError, match="Could not connect to AirSim"):
        manager.connect(retries=1, retry_delay_s=0.0)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert manager.state.connected is False


def test_confirm_connection_with_timeout_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_airsim = type("FakeAirSim", (), {"MultirotorClient": _HangingClient})
    monkeypatch.setattr(client_module, "airsim", fake_airsim)
    manager = AirSimConnectionManager(AirSimConfig(timeout_s=0.1))
    client = fake_airsim.MultirotorClient()  # type: ignore[attr-defined]

    with pytest.raises(TimeoutError, match="confirmConnection timed out"):
        manager._confirm_connection_with_timeout(client)

    # The worker thread is daemonized, so the test process can continue safely.
    assert any(thread.name == "AirSimConfirmConnection" for thread in threading.enumerate())
