import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO)


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates a trained classification model and visualizes performance metrics.

    This function performs:
    - Classification report (precision, recall, F1-score)
    - Confusion matrix visualization with per-cell counts
    - ROC AUC (macro and micro averages) if the model supports probability estimates
    - ROC curve (micro-average)

    Parameters:
    -----------
    model : sklearn-like classifier
        A trained classification model implementing .predict() and optionally .predict_proba().

    X_test : pd.DataFrame or np.ndarray
        Feature matrix for the test set.

    y_test : array-like
        True labels for the test set.

    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder used to transform the target variable. Used to recover original class names.

    Returns:
    --------
    None
        Displays plots and logs evaluation metrics using the logging module.
    """
    y_pred = model.predict(X_test)
    logging.info(
        "\n"
        + classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    )

    cm = confusion_matrix(y_test, y_pred)

    # Plot com matplotlib puro
    fig, ax = plt.subplots(figsize=(10, 5.625))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(label_encoder.classes_)),
        yticks=np.arange(len(label_encoder.classes_)),
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Números visíveis manualmente
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.show()

    # ROC AUC (Macro/Micro)
    y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
        roc_auc_macro = roc_auc_score(
            y_test_bin, y_score, average="macro", multi_class="ovr"
        )
        roc_auc_micro = roc_auc_score(
            y_test_bin, y_score, average="micro", multi_class="ovr"
        )

        logging.info(f"ROC AUC (macro-average): {roc_auc_macro:.4f}")
        logging.info(f"ROC AUC (micro-average): {roc_auc_micro:.4f}")

        # ROC Curve (micro-average)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        plt.figure(figsize=(10, 5.625))
        plt.plot(
            fpr,
            tpr,
            color="blue",
            label=f"Micro-average ROC (AUC = {roc_auc_micro:.2f})",
        )
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Micro-average)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        logging.warning(
            "Model does not support probability estimates; ROC AUC not available."
        )
