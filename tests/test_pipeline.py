import matplotlib
import pandas as pd

from src.evaluation import evaluate_model
from src.modeling import train_decision_tree
from src.preprocessing import preprocess_data

matplotlib.use("Agg")


def test_model_pipeline():
    df = pd.read_csv("data/root_cause.csv")
    X, y, le = preprocess_data(df)

    # Split (very basic)
    X_train, X_test = X.iloc[:800], X.iloc[800:]
    y_train, y_test = y.iloc[:800], y.iloc[800:]

    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test, le)
