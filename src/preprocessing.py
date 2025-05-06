import pickle
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


def preprocess_data(
    df: pd.DataFrame,
    remove_corr: bool = True,
    apply_sampling: bool = False,
    save_path=f"outputs/encoders/{int(time.time())}_label_encoder.pkl",
):
    """
    Preprocesses the input DataFrame for machine learning.

    This function performs the following steps:
    - Drops redundant columns such as 'Unnamed: 0' and 'ID'
    - Ensures binary error columns are of integer type
    - Encodes the target variable 'issue_found' using label encoding
    - Selects features that start with 'value_' or 'error_' prefixes
    - Optionally removes highly correlated features
    - Optionally applies SMOTE oversampling to balance classes

    Parameters:
    -----------
    df : pd.DataFrame
        The raw input DataFrame containing features and the target column.
    remove_corr : bool
        Whether to remove highly correlated features (default: True).
    apply_sampling : bool
        Whether to apply SMOTE sampling for class balancing (default: False).

    Returns:
    --------
    X : pd.DataFrame or np.ndarray
        The preprocessed feature matrix (resampled if sampling is applied).

    y : pd.Series or np.ndarray
        The encoded target variable (resampled if sampling is applied).

    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder fitted on the target variable.
    """
    df = df.copy()

    # Drop redundant columns
    redundant_cols = [col for col in df.columns if col in ["Unnamed: 0", "ID"]]
    df.drop(columns=redundant_cols, inplace=True, errors="ignore")

    # Ensure binary error columns are integers
    error_cols = [col for col in df.columns if col.startswith("error_")]
    df.loc[:, error_cols] = df[error_cols].astype(int)

    # Encode target
    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["issue_found"])
    with open(save_path, "wb") as f:
        pickle.dump(label_encoder, f)

    # Select features
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("value_") or col.startswith("error_")
    ]
    X = df[feature_cols]

    # Remove highly correlated features if requested
    if remove_corr:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X = X.drop(columns=to_drop)

    y = df["target"]

    # Apply SMOTE sampling if requested
    if apply_sampling:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    return X, y, label_encoder
