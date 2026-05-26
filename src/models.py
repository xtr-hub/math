"""Modeling utilities."""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics for a fitted model."""
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        print(classification_report(y_test, preds))
    else:
        mse = mean_squared_error(y_test, preds)
        print(f"MSE: {mse:.4f}")
    return preds
