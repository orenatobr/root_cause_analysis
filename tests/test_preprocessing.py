import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocess_data():
    data = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "ID": [100, 101],
            "value_1": [0.5, 0.9],
            "value_2": [0.1, 0.2],
            "error_1": [0, 1],
            "issue_found": ["CPU_OVERHEAT", "MEMORY_LEAK"],
        },
    )
    X, y, le = preprocess_data(data)
    print(X.shape)

    assert "Unnamed: 0" not in X.columns
    assert "ID" not in X.columns
    assert X.shape[1] == 1
    assert y.tolist() == [
        le.transform(["CPU_OVERHEAT"])[0],
        le.transform(["MEMORY_LEAK"])[0],
    ]
