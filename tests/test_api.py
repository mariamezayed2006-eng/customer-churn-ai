from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert 0 <= body["churn_probability"] <= 1
        assert body["risk_level"] in ("Low", "Medium", "High")


def test_invalid_input():
    invalid = VALID_CUSTOMER.copy()
    invalid["SeniorCitizen"] = 3
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
