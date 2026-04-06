# Diabetes ML service — run FastAPI with uvicorn.
# Prerequisites: pip install -e ., trained model at MODEL_PATH (default under models/).
# Working directory: ml-service root.
# Listens on 0.0.0.0:8000 (health /predict).

Set-Location $PSScriptRoot\..
python -m pip install -e . -q
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
