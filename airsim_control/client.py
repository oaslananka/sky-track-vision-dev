from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from config.settings import AirSimConfig

logger = logging.getLogger("skytrackvision.airsim.client")

try:
    import airsim
except Exception:  # pragma: no cover - optional runtime dependency
    airsim = None


@dataclass(slots=True)
class VehicleConnectionState:
    connected: bool
    api_control_enabled: bool
    timestamp_ns: int


def get_client_rpc_lock(client: Any) -> threading.RLock:
    """Return a process-local RPC lock attached to the AirSim client instance."""
    lock = getattr(client, "_skytrackvision_rpc_lock", None)
    if lock is None:
        lock = threading.RLock()
        client._skytrackvision_rpc_lock = lock
    return lock


class AirSimConnectionManager:
    """Own the AirSim client lifecycle and expose a narrow connection surface."""

    def __init__(self, cfg: AirSimConfig) -> None:
        self._cfg = cfg
        self._client: Any | None = None
        self._state = VehicleConnectionState(False, False, time.time_ns())

    @property
    def client(self) -> Any:
        if self._client is None:
            raise ConnectionError("AirSim client is not connected")
        return self._client

    @property
    def state(self) -> VehicleConnectionState:
        return self._state

    def _confirm_connection_with_timeout(self, client: Any) -> None:
        result: dict[str, Any] = {}
        finished = threading.Event()

        def _worker() -> None:
            try:
                client.confirmConnection()
                result["ok"] = True
            except Exception as exc:
                result["exc"] = exc
            finally:
                finished.set()

        thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="AirSimConfirmConnection",
        )
        thread.start()
        if not finished.wait(self._cfg.timeout_s):
            raise TimeoutError(
                f"AirSim confirmConnection timed out after {self._cfg.timeout_s:.1f}s"
            )
        exc = result.get("exc")
        if exc is not None:
            raise exc

    def connect(self, retries: int = 5, retry_delay_s: float = 3.0) -> Any:
        if airsim is None:
            raise ConnectionError("airsim package is not installed")
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                client = airsim.MultirotorClient(ip=self._cfg.host, port=self._cfg.port)
                self._confirm_connection_with_timeout(client)
                client.enableApiControl(True, vehicle_name=self._cfg.vehicle_name)
                client.armDisarm(True, vehicle_name=self._cfg.vehicle_name)
                self._client = client
                self._state = VehicleConnectionState(True, True, time.time_ns())
                return self._client
            except Exception as exc:
                last_exc = exc
                self._client = None
                self._state = VehicleConnectionState(False, False, time.time_ns())
                logger.warning(
                    "AirSim connection attempt %s/%s failed: %s",
                    attempt,
                    retries,
                    exc,
                )
                if attempt < retries:
                    time.sleep(retry_delay_s)
        raise ConnectionError(
            f"Could not connect to AirSim at {self._cfg.host}:{self._cfg.port} "
            f"after {retries} attempts"
        ) from last_exc

    def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            self._client.armDisarm(False, vehicle_name=self._cfg.vehicle_name)
            self._client.enableApiControl(False, vehicle_name=self._cfg.vehicle_name)
        except Exception:
            pass
        self._state = VehicleConnectionState(False, False, time.time_ns())
        self._client = None

    def is_connected(self) -> bool:
        return self._client is not None and self._state.connected
