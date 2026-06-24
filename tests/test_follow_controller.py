from __future__ import annotations

from autonomy.contracts import MotionIntent, MotionPrimitive
from autonomy.follow_controller import FollowController
from autonomy.ibvs import IBVSController
from config.settings import IBVSConfig, PilotConfig
from tests.conftest import make_snapshot, make_target, make_telemetry


def test_scan_intent_generates_roaming_motion() -> None:
    controller = FollowController(IBVSController(IBVSConfig()), PilotConfig())

    cmd = controller.resolve(
        MotionIntent(primitive=MotionPrimitive.SCAN),
        target=None,
        snapshot=make_snapshot(),
        telemetry=make_telemetry(),
        frame_w=640,
        frame_h=480,
    )

    assert cmd.source == "scan"
    assert cmd.vx > 0.0
    assert abs(cmd.yaw_rate) > 0.0


def test_orbit_without_target_stays_in_motion() -> None:
    controller = FollowController(IBVSController(IBVSConfig()), PilotConfig())

    cmd = controller.resolve(
        MotionIntent(primitive=MotionPrimitive.ORBIT),
        target=None,
        snapshot=make_snapshot(),
        telemetry=make_telemetry(),
        frame_w=640,
        frame_h=480,
    )

    assert cmd.source == "orbit_search"
    assert cmd.vy > 0.2


def test_follow_confirmed_target_uses_ibvs_source() -> None:
    controller = FollowController(IBVSController(IBVSConfig()), PilotConfig())

    cmd = controller.resolve(
        MotionIntent(primitive=MotionPrimitive.FOLLOW, target_id=7),
        target=make_target(),
        snapshot=make_snapshot(),
        telemetry=make_telemetry(),
        frame_w=640,
        frame_h=480,
    )

    assert cmd.source == "ibvs"
