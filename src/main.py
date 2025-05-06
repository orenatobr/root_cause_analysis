import logging
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from data_loader import load_data
from evaluation import compare_models_statistically, evaluate_model
from modeling import train_decision_tree, train_xgboost
from preprocessing import preprocess_data
from utils import suggest_action

# ==============================
# Configuration
# ==============================
logging.basicConfig(level=logging.INFO)
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

app = FastAPI(title="Root Cause Analysis Inference API")


# ==============================
# Core Functions
# ==============================
def training():
    """Executes the training pipeline."""
    data_path = Path("data/root_cause.csv")
    df = load_data(data_path)
    X, y, label_encoder = preprocess_data(df, remove_corr=False, apply_sampling=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42,
    )

    dt_model = train_decision_tree(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    evaluate_model(dt_model, X_test, y_test, label_encoder)
    evaluate_model(xgb_model, X_test, y_test, label_encoder)

    models = {"Decision Tree": dt_model, "XGBoost": xgb_model}
    compare_models_statistically(models, X_test, y_test, scoring="f1_weighted")


def test_inference_endpoint():
    """Sends a sample payload to the inference endpoint for testing."""
    url = "http://localhost:8080/predict"
    payload = {
        "value_1": 2.033852,
        "value_2": 0.860043,
        "value_3": 0.839214,
        "value_4": 0.876370,
        "error_1": 1,
        "error_2": 0,
        "error_3": 1,
        "error_4": 0,
        "error_5": 0,
        "error_6": 0,
        "error_7": 1,
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        logging.info("✅ Success!")
        logging.info(f"Prediction: {response.json()}")
    else:
        logging.warning(f"❌ Failed with status code {response.status_code}")
        logging.warning(f"Response: {response.text}")


def get_latest_file(folder_path):
    """Returns the most recently modified file in a folder."""
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return max(files, key=os.path.getmtime) if files else None


# ==============================
# Inference API Setup
# ==============================
class InputData(BaseModel):
    value_1: float
    value_2: float
    value_3: float
    value_4: float
    error_1: int
    error_2: int
    error_3: int
    error_4: int
    error_5: int
    error_6: int
    error_7: int


@app.post("/predict")
def predict(data: InputData):
    """Returns predicted label and confidence for input payload."""
    try:
        model = joblib.load(get_latest_file("outputs/models"))
        label_encoder = joblib.load(get_latest_file("outputs/encoders"))

        try:
            expected_columns = model.get_booster().feature_names
        except AttributeError:
            expected_columns = model.feature_names_in_

        df = pd.DataFrame([data.model_dump()])
        pred = model.predict(df[expected_columns])[0]
        label = label_encoder.inverse_transform([pred])[0]
        action = suggest_action(pred, label_encoder)
        proba = model.predict_proba(df[expected_columns]).max()

        return {
            "predicted_label": label,
            "confidence": round(float(proba), 4),
            "action": action,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    step = os.getenv("STEP")
    if step == "training":
        training()
    elif step == "inference":
        uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
    elif step == "test":
        test_inference_endpoint()
