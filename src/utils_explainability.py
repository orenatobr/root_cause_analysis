import numpy as np


def get_prediction_confidence(model, X_input):
    """
    Returns prediction with confidence bucket.

    Parameters
    ----------
    model : fitted classifier
    X_input : pd.DataFrame or np.ndarray

    Returns
    -------
    Tuple(label, confidence_score, confidence_label)
    """
    proba = model.predict_proba(X_input)
    confidence = np.max(proba, axis=1)
    preds = np.argmax(proba, axis=1)
    label = []

    for conf in confidence:
        if conf >= 0.9:
            label.append("High")
        elif conf >= 0.6:
            label.append("Medium")
        else:
            label.append("Low")
    return preds, confidence, label


SUGGESTED_ACTIONS = {
    "CPU_OVERHEAT": "Check fan speed and thermal paste.",
    "MEMORY_LEAK": "Restart memory-intensive services.",
    "NETWORK_DELAY": "Inspect network load and latency sources.",
    "DATABASE_ISSUE": "Reindex tables and check DB memory limits.",
    "POWER_OFF": "Validate power supply and UPS system.",
}


def suggest_action(label, label_encoder):
    """
    Suggest preventive action given model output label.

    Parameters
    ----------
    label : int
        Encoded prediction.
    label_encoder : fitted LabelEncoder

    Returns
    -------
    str
        Suggested action string.
    """
    decoded = label_encoder.inverse_transform([label])[0]
    return SUGGESTED_ACTIONS.get(decoded, "No action defined.")
