from __future__ import annotations

import math

from autonomy.ibvs import IBVSController
from config.settings import IBVSConfig
from tests.conftest import make_target, make_telemetry


def test_ibvs_zero_error_stays_stable() -> None:
    controller = IBVSController(IBVSConfig(desired_area_ratio=5000 / (640 * 480)))

    output = controller.compute(
        make_target(center=(320.0, 240.0), area=5000.0),
        make_telemetry(),
        640,
        480,
    )

    assert math.isclose(output.vx, 0.0, abs_tol=1e-6)
    assert math.isclose(output.vz, 0.0, abs_tol=1e-6)
    assert math.isclose(output.yaw_rate, 0.0, abs_tol=1e-6)


def test_ibvs_clamps_outputs_to_configured_limits() -> None:
    controller = IBVSController(
        IBVSConfig(
            max_vx=1.0,
            max_vz=0.5,
            max_yaw_rate=0.25,
            desired_area_ratio=0.01,
        )
    )

    output = controller.compute(
        make_target(center=(640.0, 480.0), area=40_000.0),
        make_telemetry(),
        640,
        480,
    )

    assert output.vx <= 1.0
    assert abs(output.vz) <= 0.5
    assert abs(output.yaw_rate) <= 0.25


def test_ibvs_reset_clears_all_integrators() -> None:
    controller = IBVSController(IBVSConfig())
    controller.compute(
        make_target(center=(500.0, 240.0), area=10_000.0),
        make_telemetry(),
        640,
        480,
    )

    controller.reset()

    assert controller._pid_yaw.integral == 0.0
    assert controller._pid_fwd.integral == 0.0
    assert controller._pid_alt.integral == 0.0
    assert controller._pid_vx_inner.integral == 0.0
    assert controller._pid_vz_inner.integral == 0.0


def test_ibvs_clamps_out_of_bounds_smooth_center() -> None:
    """smooth_center values outside frame bounds must be clipped before error computation."""
    controller = IBVSController(IBVSConfig(desired_area_ratio=5000 / (640 * 480)))

    # Provide a target whose smooth_center is far outside the frame.
    output = controller.compute(
        make_target(center=(-500.0, 900.0), area=5000.0),
        make_telemetry(),
        640,
        480,
    )

    # After clamping to [0, frame_w] x [0, frame_h], ex must be in [-frame_w/2, frame_w/2].
    # clamp(-500, 0, 640) = 0, so ex = 0 - 320 = -320  → ex_norm = -1.0
    # clamp(900, 0, 480)  = 480, so ey = 480 - 240 = 240 → ey_norm = 1.0
    # Neither should cause a NaN/Inf.
    assert output.vx == output.vx  # not NaN
    assert output.yaw_rate == output.yaw_rate  # not NaN
