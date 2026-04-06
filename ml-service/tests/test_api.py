"""
HTTP tests against the FastAPI app using TestClient.

The model loaded at startup is the conftest dummy joblib (fixed probabilities).
"""


def test_health(client):
    """GET /health returns 200 and status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_success(client):
    body = {
        "pregnancies": 1,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 20,
        "insulin": 80,
        "bmi": 28,
        "diabetes_pedigree_function": 0.5,
        "age": 30,
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "outcome" in data
    assert "probability_no_diabetes" in data
    assert "probability_diabetes" in data


def test_predict_422_missing_bmi(client):
    """Missing BMI and Nutritional_Status yields 422 from validation layer."""
    body = {
        "pregnancies": 1,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 20,
        "insulin": 80,
        "diabetes_pedigree_function": 0.5,
        "age": 30,
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 422
