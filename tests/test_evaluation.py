import logging

import matplotlib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.evaluation import evaluate_model

matplotlib.use("Agg")


def test_evaluate_model_runs_without_error(caplog):
    caplog.set_level(logging.INFO)

    # Criar dataset artificial
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=3,
        n_informative=3,
        random_state=42,
    )
    labels = np.array(["CPU_OVERHEAT", "DATABASE_ISSUE", "NETWORK_DELAY"])
    y = labels[y]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
    )

    # Modelo simples
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Executa avaliação
    evaluate_model(model, X_test, y_test, le)

    # Verifica que métricas foram logadas
    assert "precision" in caplog.text or "ROC AUC" in caplog.text
