import logging

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)


def train_decision_tree(X_train, y_train, fine_tune=False):
    """
    Trains a Decision Tree classifier, optionally using hyperparameter tuning.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : array-like
        Target values for training.
    fine_tune : bool
        Whether to perform GridSearchCV hyperparameter tuning.

    Returns:
    --------
    model : DecisionTreeClassifier
        A trained (and optionally fine-tuned) Decision Tree model.
    """
    if fine_tune:
        param_grid = {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Decision Tree params: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train, fine_tune=False):
    """
    Trains an XGBoost classifier, optionally using hyperparameter tuning.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    y_train : array-like
        Target values for training.
    fine_tune : bool
        Whether to perform GridSearchCV hyperparameter tuning.

    Returns:
    --------
    model : XGBClassifier
        A trained (and optionally fine-tuned) XGBoost model.
    """
    if fine_tune:
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        grid_search = GridSearchCV(
            XGBClassifier(eval_metric="mlogloss", random_state=42),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Best XGBoost params: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = XGBClassifier(eval_metric="mlogloss", random_state=42)
        model.fit(X_train, y_train)

    return model
