"""
Evaluation module for California Housing models.
Calculates metrics (MAE, RMSE, R2), generates visual diagnostic plots,
and supports evaluating MLflow registered models or local models.
"""
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculate key regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2),
    }


def plot_residuals(y_true, y_pred, output_path: str = "reports/figures/residuals_plot.png"):
    """Generate and save residual plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, color="royalblue", edgecolors="none")
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Value ($100k)")
    plt.ylabel("Residuals ($100k)")
    plt.title("Residuals vs Predicted Values")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_actual_vs_predicted(y_true, y_pred, output_path: str = "reports/figures/actual_vs_predicted.png"):
    """Generate and save Actual vs Predicted scatter plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color="forestgreen", edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=1.5)
    plt.xlabel("Actual Value ($100k)")
    plt.ylabel("Predicted Value ($100k)")
    plt.title("Actual vs Predicted House Values")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_feature_importance(model, feature_names, output_path: str = "reports/figures/feature_importance.png"):
    """Plot feature importances or coefficients if available."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importances"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        title = "Absolute Feature Coefficients"
    else:
        return None

    indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align="center", color="teal")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def evaluate_model_uri(model_uri: str, data_path: str = "data/dataset.csv", target_col: str = "MedHouseVal"):
    """Load model from MLflow URI and evaluate on the dataset."""
    import mlflow
    import mlflow.pyfunc

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    print(f"Evaluation results for '{model_uri}': {json.dumps(metrics, indent=2)}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model-uri", help="MLflow model URI (e.g. models:/HousePriceModel@production)")
    parser.add_argument("--data-path", default="data/dataset.csv", help="Path to evaluation dataset")
    args = parser.parse_args()

    if args.model_uri:
        evaluate_model_uri(args.model_uri, args.data_path)
