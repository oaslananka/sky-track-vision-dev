$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install --no-build-isolation airsim
python -m pip install -r requirements-dev.txt
python .\main.py @args

