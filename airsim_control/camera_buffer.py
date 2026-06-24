"""Thread-safe camera frame buffer for non-blocking frame acquisition.

Decouples AirSim TCP image fetch (150-250ms) from the main HUD loop so that
the display and control ticks are not blocked by network latency.
"""

from __future__ import annotations

import logging
import threading
import time

from autonomy.contracts import FramePacket

logger = logging.getLogger("skytrackvision.airsim_control.camera_buffer")


class CameraBuffer:
    """Thread-safe, drop-old-frames single-slot buffer.

    The producer (CameraThread) overwrites the latest frame; the consumer
    (PilotDisplay._run_loop) reads it.  Old frames are silently dropped so
    the consumer always sees the most recent image.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._event = threading.Event()

    def put(self, packet: FramePacket) -> None:
        """Store a new frame, overwriting any unread previous frame."""
        with self._lock:
            self._latest = packet
        self._event.set()

    def get(self, timeout: float = 0.05) -> FramePacket | None:
        """Return the latest frame, blocking at most *timeout* seconds.

        Returns ``None`` if no frame arrives within the timeout window.
        """
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._lock:
            return self._latest


class CameraThread(threading.Thread):
    """Daemon thread that continuously fetches frames and pushes to a buffer."""

    def __init__(self, camera: object, buf: CameraBuffer) -> None:
        super().__init__(daemon=True, name="CameraFetch")
        self._camera = camera
        self._buf = buf
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                packet = self._camera.get_frame()  # type: ignore[attr-defined]
                self._buf.put(packet)
            except Exception:
                # Avoid tight-loop on persistent errors
                time.sleep(0.01)

    def stop(self) -> None:
        """Signal the thread to exit (will join on next iteration)."""
        self._running = False
