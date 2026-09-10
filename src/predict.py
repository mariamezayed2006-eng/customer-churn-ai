from pathlib import Path
from typing import Any
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run 'python src/train.py' first."
        )
    return joblib.load(MODEL_PATH)


def predict_customer(customer: dict[str, Any]):
    model = load_model()
    frame = pd.DataFrame([customer])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk,
    }
