from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ui.recorder import RealtimeRecorder


class MockClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += delta


class MockVideoWriter:
    def __init__(self) -> None:
        self.frames: list = []
        self.opened = True
        self.released = False

    def write(self, frame: object) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.released = True

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened


def make_frame(height: int = 480, width: int = 640) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def factory(mock: MockVideoWriter):  # type: ignore[no-untyped-def]
    def _factory(path: str, fps: int, size: tuple[int, int]) -> MockVideoWriter:
        return mock

    return _factory


class TestRealtimeRecorderInit:
    def test_fps_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="fps must be positive"):
            RealtimeRecorder("out.mp4", fps=0)

    def test_fps_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="fps must be positive"):
            RealtimeRecorder("out.mp4", fps=-10)

    def test_accepts_path_or_str(self) -> None:
        r = RealtimeRecorder(Path("out.mp4"))
        assert r.path == "out.mp4"
        r.close()


class TestRealtimeRecorderRecording:
    def test_writer_opened_on_first_frame(self) -> None:
        mock_writer = MockVideoWriter()
        clock = MockClock()
        r = RealtimeRecorder("out.mp4", writer_factory=factory(mock_writer), clock=clock)
        frame = make_frame()
        r.add(frame)
        assert r.frames_written == 1
        assert mock_writer.frames == [frame]
        r.close()
        assert mock_writer.released

    def test_wall_clock_duplicates_catch_up_frames(self) -> None:
        mock_writer = MockVideoWriter()
        clock = MockClock()
        r = RealtimeRecorder("out.mp4", fps=30, writer_factory=factory(mock_writer), clock=clock)
        r.add(make_frame())
        assert r.frames_written == 1
        clock.advance(0.1)
        r.add(make_frame())
        assert r.frames_written == 4
        r.close()

    def test_catchup_capped_at_max_frames(self) -> None:
        mock_writer = MockVideoWriter()
        clock = MockClock()
        r = RealtimeRecorder("out.mp4", fps=30, writer_factory=factory(mock_writer), clock=clock)
        r.add(make_frame())
        clock.advance(10.0)
        r.add(make_frame())
        assert r.frames_written == 1 + RealtimeRecorder._MAX_CATCHUP_FRAMES
        r.close()

    def test_writer_failure_raises_runtime_error(self) -> None:
        mock_writer = MockVideoWriter()
        mock_writer.opened = False
        r = RealtimeRecorder("out.mp4", writer_factory=factory(mock_writer), clock=MockClock())
        with pytest.raises(RuntimeError, match="failed to open"):
            r.add(make_frame())

    def test_frame_size_change_raises_value_error(self) -> None:
        r = RealtimeRecorder(
            "out.mp4",
            writer_factory=factory(MockVideoWriter()),
            clock=MockClock(),
        )
        r.add(make_frame(480, 640))
        with pytest.raises(ValueError, match="Frame size changed"):
            r.add(make_frame(720, 1280))
        r.close()

    def test_close_releases_writer(self) -> None:
        mock_writer = MockVideoWriter()
        r = RealtimeRecorder("out.mp4", writer_factory=factory(mock_writer), clock=MockClock())
        r.add(make_frame())
        r.close()
        assert mock_writer.released
        assert not r.active

    def test_close_idempotent(self) -> None:
        mock_writer = MockVideoWriter()
        r = RealtimeRecorder("out.mp4", writer_factory=factory(mock_writer), clock=MockClock())
        r.add(make_frame())
        r.close()
        assert mock_writer.released
        r.close()
