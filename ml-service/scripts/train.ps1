# Diabetes ML service — train pipeline and save joblib.
# Prerequisites: Python 3.11+, ml-service/data/diabetes.csv (or DATA_PATH in .env).
# Working directory: changes to ml-service root (parent of scripts/).
# Runs: editable install then python -m ml_service.train (writes MODEL_PATH or OUTPUT_MODEL_PATH).

Set-Location $PSScriptRoot\..
python -m pip install -e . -q
python -m ml_service.train
