#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install --no-build-isolation airsim
python -m pip install -r requirements-dev.txt

echo "SkyTrackVision dependencies are installed."

