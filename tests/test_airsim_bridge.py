from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from airsim_control import movement as movement_module
from airsim_control.movement import DroneMovementController
from airsim_control.sensors import SensorSuiteReader
from autonomy.contracts import VelocityCmd
from autonomy.safety import SafetyEvaluator
from config.settings import AirSimConfig, SafetyConfig
from skypilot.airsim_bridge import AirSimBridge
from tests.conftest import make_snapshot


class _FakeMovement:
    def __init__(self) -> None:
        self.serial_command_active = False
        self.altitude_calls: list[tuple[float, float]] = []
        self.velocity_calls: list[VelocityCmd] = []

    def move_to_altitude(self, altitude_m: float, velocity: float = 1.0) -> None:
        self.altitude_calls.append((altitude_m, velocity))

    def hover(self) -> None:
        return None

    def move_by_velocity(self, cmd: VelocityCmd) -> None:
        self.velocity_calls.append(cmd)
        return None


class _FakeSensorReader:
    def __init__(self) -> None:
        self.last_snapshot = make_snapshot()
        self.fail_read = False
        self.read_calls = 0

    def read(self) -> object:
        self.read_calls += 1
        if self.fail_read:
            raise RuntimeError("sensor read failed")
        return self.last_snapshot


def test_move_to_altitude_passes_requested_velocity() -> None:
    movement: Any = _FakeMovement()
    sensor_reader: Any = _FakeSensorReader()
    bridge = AirSimBridge(movement, sensor_reader, SafetyEvaluator(SafetyConfig()))

    bridge.move_to_altitude(4.0, velocity=2.75)

    assert cast(_FakeMovement, bridge._movement).altitude_calls == [(4.0, 2.75)]


def test_move_falls_back_to_cached_snapshot_when_sensor_read_fails() -> None:
    movement = _FakeMovement()
    sensor_reader = _FakeSensorReader()
    bridge = AirSimBridge(
        cast(Any, movement),
        cast(Any, sensor_reader),
        SafetyEvaluator(SafetyConfig()),
    )
    sensor_reader.fail_read = True

    moved = bridge.move(VelocityCmd(0.0, 0.0, 0.0, 0.0, 0.1, "hover"))

    assert moved is True
    assert len(movement.velocity_calls) == 1


def test_move_reuses_prefetched_snapshot_without_second_sensor_read() -> None:
    movement = _FakeMovement()
    sensor_reader = _FakeSensorReader()
    bridge = AirSimBridge(
        cast(Any, movement),
        cast(Any, sensor_reader),
        SafetyEvaluator(SafetyConfig()),
    )
    sensor_reader.read_calls = 0

    moved = bridge.move(
        VelocityCmd(0.1, 0.0, 0.0, 0.0, 0.1, "scan"),
        snapshot=cast(Any, sensor_reader.last_snapshot),
    )

    assert moved is True
    assert sensor_reader.read_calls == 0
    assert len(movement.velocity_calls) == 1


class _FakeFuture:
    def __init__(self) -> None:
        self.join_called = False

    def join(self) -> None:
        self.join_called = True
        return None


class _FakeAirSimClient:
    def __init__(self) -> None:
        self.last_velocity_future: _FakeFuture | None = None
        self.last_hover_future: _FakeFuture | None = None

    def moveToZAsync(  # noqa: N802
        self,
        z: float,
        velocity: float,
        *,
        vehicle_name: str,
    ) -> _FakeFuture:
        del z, velocity, vehicle_name
        return _FakeFuture()

    def getLidarData(  # noqa: N802
        self,
        *,
        lidar_name: str,
        vehicle_name: str,
    ) -> SimpleNamespace:
        del lidar_name, vehicle_name
        return SimpleNamespace(point_cloud=[])

    def getDistanceSensorData(  # noqa: N802
        self,
        *,
        distance_sensor_name: str,
        vehicle_name: str,
    ) -> SimpleNamespace:
        del distance_sensor_name, vehicle_name
        return SimpleNamespace(distance=10.0)

    def getMultirotorState(  # noqa: N802
        self,
        *,
        vehicle_name: str,
    ) -> SimpleNamespace:
        del vehicle_name
        zero = SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0)
        orientation = SimpleNamespace(w_val=1.0, x_val=0.0, y_val=0.0, z_val=0.0)
        kinematics = SimpleNamespace(
            orientation=orientation,
            position=zero,
            linear_velocity=zero,
        )
        return SimpleNamespace(kinematics_estimated=kinematics, gps_location=object())

    def moveByVelocityAsync(  # noqa: N802
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
        *,
        yaw_mode: object,
        vehicle_name: str,
    ) -> _FakeFuture:
        del vx, vy, vz, duration, yaw_mode, vehicle_name
        future = _FakeFuture()
        self.last_velocity_future = future
        return future

    def hoverAsync(  # noqa: N802
        self,
        *,
        vehicle_name: str,
    ) -> _FakeFuture:
        del vehicle_name
        future = _FakeFuture()
        self.last_hover_future = future
        return future


def test_sensor_reader_and_movement_controller_share_same_client_rpc_lock() -> None:
    client = _FakeAirSimClient()
    cfg = AirSimConfig()

    movement = DroneMovementController(client, cfg)
    sensors = SensorSuiteReader(client, cfg)

    assert movement._rpc_lock is sensors._rpc_lock


def test_movement_controller_joins_velocity_and_hover_futures(
    monkeypatch,
) -> None:
    client = _FakeAirSimClient()
    cfg = AirSimConfig()
    monkeypatch.setattr(
        movement_module,
        "airsim",
        SimpleNamespace(YawMode=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    movement = DroneMovementController(client, cfg)

    movement.hover()
    movement.move_by_velocity(VelocityCmd(0.1, 0.0, 0.0, 0.0, 0.1, "scan"))

    assert client.last_hover_future is not None
    assert client.last_hover_future.join_called is True
    assert client.last_velocity_future is not None
    assert client.last_velocity_future.join_called is True
