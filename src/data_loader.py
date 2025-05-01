import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    -----------
    path : str
        The file path to the CSV file.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the data loaded from the CSV file.
    """
    return pd.read_csv(path)