from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from airsim_control.camera import DroneCameraStream
from config.settings import AirSimConfig


def test_camera_decodes_raw_uncompressed_frame() -> None:
    cfg = AirSimConfig(camera_compress=False)
    camera = DroneCameraStream(client=object(), cfg=cfg)
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    response = SimpleNamespace(
        image_data_uint8=rgb.tobytes(),
        width=2,
        height=2,
    )

    frame = camera._decode(response.image_data_uint8, width=response.width, height=response.height)

    assert tuple(frame[0, 0]) == (0, 0, 255)
    assert tuple(frame[0, 1]) == (0, 255, 0)
    assert tuple(frame[1, 0]) == (255, 0, 0)
    assert tuple(frame[1, 1]) == (255, 255, 255)
