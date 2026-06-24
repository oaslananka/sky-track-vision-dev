"""Wall-clock-accurate HUD recorder for the classic runtime.

The classic loop renders a composited HUD frame each tick, but its processing
rate jitters (YOLO inference, sensor reads). Feeding those frames straight to a
``cv2.VideoWriter`` at a fixed fps yields video that plays back too fast or too
slow, because the writer assumes every ``write()`` is exactly ``1/fps`` apart.

``RealtimeRecorder`` keeps playback real-time-accurate at a *constant* output
fps by duplicating the latest frame to fill wall-clock gaps — the same trick a
screen recorder uses. Recording stays on the main thread: the classic loop
already paces itself to ~30 fps (see ``main.py``), leaving ample slack for the
cheap mp4v encode, so this does not introduce display stutter.

The writer and clock are injectable so the pacing logic can be unit-tested
deterministically without a real codec.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class VideoWriterLike(Protocol):
    """The slice of ``cv2.VideoWriter`` this recorder depends on."""

    def write(self, frame: Any) -> None: ...

    def release(self) -> None: ...

    def isOpened(self) -> bool: ...  # noqa: N802


def _default_writer(path: str, fps: int, size: tuple[int, int]) -> VideoWriterLike:
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)


class RealtimeRecorder:
    """Records frames at a constant output fps regardless of loop jitter."""

    _MAX_CATCHUP_FRAMES = 5

    def __init__(
        self,
        path: str | Path,
        fps: int = 30,
        *,
        writer_factory: Callable[[str, int, tuple[int, int]], VideoWriterLike] = _default_writer,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._path = str(path)
        self._fps = fps
        self._interval = 1.0 / fps
        self._writer_factory = writer_factory
        self._clock = clock
        self._writer: VideoWriterLike | None = None
        self._start: float | None = None
        self._written = 0
        self._frame_size: tuple[int, int] | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._written

    @property
    def active(self) -> bool:
        return self._writer is not None

    def add(self, frame: Any) -> None:
        h, w = frame.shape[:2]
        if self._frame_size is not None and (w, h) != self._frame_size:
            raise ValueError(
                f"Frame size changed: expected {self._frame_size}, got ({w}, {h})"
            )
        if self._writer is None:
            self._open(frame)
        writer = self._writer
        assert writer is not None and self._start is not None

        now = self._clock()
        target = int((now - self._start) / self._interval) + 1
        catchup = 0
        while self._written < target and catchup < self._MAX_CATCHUP_FRAMES:
            writer.write(frame)
            self._written += 1
            catchup += 1
        if self._written < target:
            self._start = now - self._written * self._interval

    def _open(self, frame: Any) -> None:
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        writer = self._writer_factory(self._path, self._fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                f"VideoWriter failed to open for {self._path} "
                f"(fps={self._fps}, size=({width}, {height}))"
            )
        self._writer = writer
        self._start = self._clock()
        self._written = 0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._start = None
            self._written = 0
