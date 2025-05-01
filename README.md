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
- Random Forest
- XGBoost with SHAP-based interpretability

## Dependencies

See `pyproject.toml` for details.

## Author

Developed as part of the selection process for the Senior AI Engineer role at OutSystems.