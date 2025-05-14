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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

app = FastAPI(title="Root Cause Analysis Inference API")

# Load data and preprocess for reuse
DATA_PATH = Path("data/root_cause.csv")
df = load_data(DATA_PATH)
X, y, label_encoder = preprocess_data(df, remove_corr=False, apply_sampling=False)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42,
)


# ==============================
# Core Functions
# ==============================
def training():
    """
    Run the training pipeline: train both models, evaluate, compare, and save the best one.
    """
    dt_model = train_decision_tree(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    evaluate_model(dt_model, X_test, y_test, label_encoder)
    evaluate_model(xgb_model, X_test, y_test, label_encoder)

    models = {"Decision Tree": dt_model, "XGBoost": xgb_model}
    compare_models_statistically(models, X_test, y_test, scoring="f1_weighted")


def test_inference_endpoint():
    """
    Send a test payload to the local inference endpoint.
    """
    url = "http://localhost:8080/predict"
    payload = {
        "value_1": -0.19,
        "value_2": 354.8,
        "value_3": 1.2,
        "value_4": 535.0,
        "error_1": 0,
        "error_2": 0,
        "error_3": 0,
        "error_4": 0,
        "error_5": 0,
        "error_6": 0,
        "error_7": 0,
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        logging.info("✅ Success!")
        logging.info(f"Prediction: {response.json()}")
    else:
        logging.warning(f"❌ Failed with status code {response.status_code}")
        logging.warning(f"Response: {response.text}")


def get_latest_file(folder_path):
    """
    Get the most recently modified file in a directory.
    """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    return max(files, key=os.path.getmtime) if files else None


def is_input_valid(df_input, df_reference, tolerance=0.1):
    """
    Check if input values are within the range of training data with a margin.

    Parameters:
    -----------
    df_input : pd.DataFrame
        The input data to check.
    df_reference : pd.DataFrame
        The reference data (e.g., X_train) to compare against.
    tolerance : float
        Percentual margin allowed (default is 10%).

    Returns:
    --------
    bool
        True if all values are within range, False otherwise.
    """
    for col in df_reference.columns:
        min_val = df_reference[col].min()
        max_val = df_reference[col].max()
        margin = (max_val - min_val) * tolerance
        if not df_input[col].between(min_val - margin, max_val + margin).all():
            logging.warning(
                f"Input value for {col} is out of range: {df_input[col].values}",
            )
            return False
    return True


# ==============================
# Inference API
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
    """
    Predicts the root cause and returns label, confidence, and suggested action.
    """
    try:
        model = joblib.load(get_latest_file("outputs/models"))
        label_encoder = joblib.load(get_latest_file("outputs/encoders"))

        try:
            expected_columns = model.get_booster().feature_names
        except AttributeError:
            expected_columns = model.feature_names_in_

        df = pd.DataFrame([data.model_dump()])
        proba = model.predict_proba(df[expected_columns])
        pred = model.predict(df[expected_columns])[0]
        max_conf = proba.max()
        label = label_encoder.inverse_transform([pred])[0]
        action = suggest_action(pred, label_encoder)

        if max_conf < 0.6 or not is_input_valid(df, X_train):
            label = "UNKNOWN"
            action = "UNKNOWN"

        return {
            "predicted_label": label,
            "confidence": round(float(max_conf), 4),
            "action": action,
        }

    except Exception as e:
        logging.exception("Inference error:")
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
