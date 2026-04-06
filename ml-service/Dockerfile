# Diabetes ML API image: install package + run uvicorn as non-root user.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY api ./api

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Drop privileges after install
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# Note: mount or bake MODEL_PATH — image has no bundled joblib by default.
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
