# Root Cause Analysis – OutSystems Challenge

This project addresses the root cause analysis challenge proposed by OutSystems for the Senior AI Engineer position.

## Objectives

- Identify combinations of system errors and parameters that lead to failures.
- Train models to predict the likely cause of a failure.
- Provide interpretability to support preventive actions.

## Project Structure

```
root_cause_analysis/
├── data/                    # Provided dataset
├── notebooks/
│   └── eda_and_modeling.ipynb  # Exploratory analysis and modeling
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── main.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── README.md
└── pyproject.toml
```

## How to Use

1. Install dependencies using Poetry:

```bash
poetry install --no-root
```

2. Run the notebook:

```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```

## Models

The following models were compared:
- Decision Tree (simple and interpretable)
- XGBoost with SHAP-based interpretability

## Dependencies

See `pyproject.toml` for details.

## Author

Developed as part of the selection process for the Senior AI Engineer role at OutSystems.


## CLI Usage with Poetry (Training, Inference API, Testing)

To run the project via command line, you can set the environment variable `STEP` to define the pipeline stage.

### 🚀 1. Train the models

```bash
STEP=training poetry run python src/main.py
```

This step will:
- Load and preprocess the dataset
- Train Decision Tree and XGBoost models
- Evaluate and compare using cross-validation + Wilcoxon test
- Save the best model to `outputs/models/` and label encoder to `outputs/encoders/`

---

### 🌐 2. Run the Inference API

```bash
STEP=inference poetry run python src/main.py
```

This will:
- Load the latest model and label encoder
- Start a FastAPI server at `http://localhost:8080/predict`

You can send POST requests with feature values to receive predictions.

---

### 🧪 3. Test the inference endpoint

```bash
STEP=test poetry run python src/main.py
```

This sends a sample request to the running FastAPI server and prints the prediction result and confidence score.
