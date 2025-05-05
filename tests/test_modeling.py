from sklearn.datasets import make_classification

from src.modeling import train_decision_tree, train_xgboost


def test_train_decision_tree_returns_fitted_model():
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    model = train_decision_tree(X, y)
    assert hasattr(model, "predict")
    assert model.tree_ is not None
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_xgboost_returns_fitted_model():
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    model = train_xgboost(X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(y)
