from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the input DataFrame for machine learning.

    This function performs the following steps:
    - Drops redundant columns such as 'Unnamed: 0' and 'ID'
    - Ensures binary error columns are of integer type
    - Encodes the target variable 'issue_found' using label encoding
    - Selects features that start with 'value_' or 'error_' prefixes

    Parameters:
    -----------
    df : pd.DataFrame
        The raw input DataFrame containing features and the target column.

    Returns:
    --------
    X : pd.DataFrame
        The preprocessed feature matrix.

    y : pd.Series
        The encoded target variable.

    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder fitted on the target variable.
    """
    df = df.copy()

    # Drop redundant columns
    redundant_cols = [col for col in df.columns if col in ["Unnamed: 0", "ID"]]
    df.drop(columns=redundant_cols, inplace=True, errors="ignore")

    # Ensure binary error columns are integers
    error_cols = [col for col in df.columns if col.startswith("error_")]
    df[error_cols] = df[error_cols].astype(int)

    # Encode target
    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["issue_found"])

    # Select feature columns
    feature_cols = [col for col in df.columns if col.startswith("value_") or col.startswith("error_")]
    X = df[feature_cols]
    y = df["target"]

    return X, y, label_encoder