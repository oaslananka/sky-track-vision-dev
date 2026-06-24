from __future__ import annotations

from config.settings import load_app_config
from vision.tracker import KalmanTracker


def main() -> None:
    cfg = load_app_config("pilot.yaml")
    tracker = KalmanTracker(cfg.vision)
    print("config_ok", cfg.mission_mode)
    print("tracker_ok", tracker is not None)


if __name__ == "__main__":
    main()
