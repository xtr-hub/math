"""Modeling utilities."""

from sklearn.metrics import classification_report, mean_squared_error


def evaluate_model(model, X_test, y_test):
    """Print evaluation metrics for a fitted model."""
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        print(classification_report(y_test, preds))
    else:
        mse = mean_squared_error(y_test, preds)
        print(f"MSE: {mse:.4f}")
    return preds
