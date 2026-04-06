# Diabetes ML service — run pytest suite.
# Prerequisites: Python 3.11+, dev/test deps from requirements / editable install.
# Working directory: ml-service root. conftest sets a dummy MODEL_PATH automatically.

Set-Location $PSScriptRoot\..
python -m pip install -e . -q
python -m pytest tests -q
