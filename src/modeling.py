from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree classifier on the provided training data.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.

    y_train : array-like
        Target values for training.

    Returns:
    --------
    model : DecisionTreeClassifier
        A fitted Decision Tree model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier on the provided training data.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.

    y_train : array-like
        Target values for training.

    Returns:
    --------
    model : XGBClassifier
        A fitted XGBoost model with multi-class log-loss as evaluation metric.
    """
    model = XGBClassifier(eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)
    return model
