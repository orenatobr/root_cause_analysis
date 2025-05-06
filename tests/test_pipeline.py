import matplotlib
import pandas as pd

from src.evaluation import evaluate_model
from src.modeling import train_decision_tree
from src.preprocessing import preprocess_data

matplotlib.use("Agg")


def mock_root_cause_data():
    return pd.DataFrame(
        [
            {
                "value_4": 0.848875,
                "value_3": 0.994232,
                "value_2": 0.105434,
                "value_1": -8.929385,
                "error_5": 1,
                "error_6": 0,
                "error_7": 0,
                "error_1": 1,
                "error_2": 1,
                "error_3": 1,
                "error_4": 0,
                "issue_found": "DATABASE_ISSUE",
            },
            {
                "value_4": 0.722715,
                "value_3": 0.271752,
                "value_2": 0.728403,
                "value_1": 1.416320,
                "error_5": 0,
                "error_6": 0,
                "error_7": 1,
                "error_1": 1,
                "error_2": 0,
                "error_3": 1,
                "error_4": 1,
                "issue_found": "NETWORK_DELAY",
            },
            {
                "value_4": 0.888405,
                "value_3": 0.059510,
                "value_2": 0.402041,
                "value_1": 3.934289,
                "error_5": 0,
                "error_6": 0,
                "error_7": 1,
                "error_1": 1,
                "error_2": 1,
                "error_3": 1,
                "error_4": 0,
                "issue_found": "CPU_OVERHEAT",
            },
        ],
    )


def test_model_pipeline():
    df = mock_root_cause_data()
    X, y, le = preprocess_data(df)

    model = train_decision_tree(X, y)
    evaluate_model(model, X, y, le)
