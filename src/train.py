"""
Training pipeline integrating Git, DVC, and MLflow for California Housing.
Implements:
- Hyperparameter tuning across 3 candidate models
- Autologging + custom metric calculation (MAE, RMSE, R2)
- Artifact logging (actual-vs-predicted, residuals, feature importance, eval report)
- Explicit model signature & input example
- Model registry registration & lifecycle promotion
- Git commit & DVC data hash audit tags
"""
import argparse
import json
import os
import subprocess
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from evaluate import evaluate_predictions, plot_actual_vs_predicted, plot_feature_importance, plot_residuals


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_sha() -> str:
    """Retrieve current Git commit SHA."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return sha
    except Exception:
        return "git-commit-unknown"


def get_dvc_data_hash(dvc_file: str = "data/dataset.csv.dvc") -> str:
    """Extract md5 hash of data file from .dvc metadata."""
    if os.path.exists(dvc_file):
        try:
            with open(dvc_file, "r") as f:
                dvc_meta = yaml.safe_load(f)
                return dvc_meta["outs"][0]["md5"]
        except Exception:
            return "dvc-parse-error"
    return "no-dvc-tracking-file"


def build_model(model_type: str, params: dict):
    """Instantiate model with standard scaler pipeline for regression."""
    if model_type == "Ridge":
        estimator = Ridge(alpha=params.get("alpha", 1.0), random_state=params.get("random_state", 42))
        return Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
    elif model_type == "RandomForestRegressor":
        estimator = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=params.get("random_state", 42),
            n_jobs=-1,
        )
        return Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
    elif model_type == "GradientBoostingRegressor":
        estimator = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.08),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
        )
        return Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_candidate(candidate_key: str, cfg: dict, register: bool = True):
    """Train a single candidate model variant, log with MLflow, and optionally register."""
    data_cfg = cfg["data"]
    mlflow_cfg = cfg["mlflow"]
    candidate_cfg = cfg["candidates"][candidate_key]

    # Setup MLflow tracking
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Load dataset
    data_path = data_cfg["raw_path"]
    target_col = data_cfg["target_column"]
    df = pd.read_csv(data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_cfg["test_size"], random_state=data_cfg["random_state"]
    )

    model_type = candidate_cfg["model_type"]
    model_pipeline = build_model(model_type, candidate_cfg)

    # Audit metadata for Git and DVC
    git_sha = get_git_commit_sha()
    dvc_hash = get_dvc_data_hash()

    run_name = f"candidate_{candidate_key}"
    print(f"\n==========================================")
    print(f"Starting MLflow Run: {run_name}")
    print(f"Model Type: {model_type} | Git SHA: {git_sha[:8]} | DVC Hash: {dvc_hash[:8]}")
    print(f"==========================================")

    # Enable autologging with exclusive/specific settings
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1. Train model
        model_pipeline.fit(X_train, y_train)

        # 2. Evaluate predictions on train & test
        y_train_pred = model_pipeline.predict(X_train)
        y_test_pred = model_pipeline.predict(X_test)

        test_metrics = evaluate_predictions(y_test, y_test_pred)
        train_metrics = evaluate_predictions(y_train, y_train_pred)

        print(f"[{candidate_key}] Test Metrics -> RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2_score']:.4f}")

        # 3. Log explicit test metrics & hyperparameters
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("rmse", test_metrics["rmse"])
        mlflow.log_metric("r2_score", test_metrics["r2_score"])

        for param_k, param_v in candidate_cfg.items():
            mlflow.log_param(param_k, param_v)

        # 4. Part C Traceability Tags
        mlflow.set_tag("git_commit_sha", git_sha)
        mlflow.set_tag("dvc_dataset_hash", dvc_hash)
        mlflow.set_tag("dataset_version_path", data_path)
        mlflow.set_tag("model_variant", candidate_key)
        mlflow.set_tag("dataset_rows", str(len(df)))

        # 5. Generate and Log Visual Artifacts
        figures_dir = os.path.join("reports", "figures", candidate_key)
        os.makedirs(figures_dir, exist_ok=True)

        res_path = plot_residuals(y_test, y_test_pred, os.path.join(figures_dir, "residuals_plot.png"))
        act_path = plot_actual_vs_predicted(y_test, y_test_pred, os.path.join(figures_dir, "actual_vs_predicted.png"))
        feat_path = plot_feature_importance(
            model_pipeline.named_steps["regressor"],
            X.columns.tolist(),
            os.path.join(figures_dir, "feature_importance.png")
        )

        mlflow.log_artifact(res_path, artifact_path="diagnostic_plots")
        mlflow.log_artifact(act_path, artifact_path="diagnostic_plots")
        if feat_path and os.path.exists(feat_path):
            mlflow.log_artifact(feat_path, artifact_path="diagnostic_plots")

        # Evaluation report artifact
        eval_report = {
            "candidate": candidate_key,
            "model_type": model_type,
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "git_commit": git_sha,
            "dvc_hash": dvc_hash,
            "dataset_rows": len(df),
        }
        report_path = os.path.join(figures_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(eval_report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # 6. Part A4: Infer Model Signature and Prepare Input Example
        input_example = X_train.head(5)
        signature = infer_signature(X_train, model_pipeline.predict(X_train))

        # Log Model with Signature & Input Example
        model_artifact_path = "model"
        registered_model_name = mlflow_cfg["registered_model_name"] if register else None

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path=model_artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        print(f"Logged model with signature:\nInputs: {signature.inputs}\nOutputs: {signature.outputs}")
        print(f"Run ID: {run_id} completed successfully.")

        return run_id, test_metrics


def manage_model_lifecycle(cfg: dict):
    """
    Apply automated promotion rules and manage lifecycle states in MLflow Model Registry.
    Demonstrates Staging/Production/Archived lifecycle and alias tagging.
    """
    mlflow_cfg = cfg["mlflow"]
    promo_cfg = cfg["promotion"]
    model_name = mlflow_cfg["registered_model_name"]

    client = MlflowClient(tracking_uri=mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    registered_versions = client.search_model_versions(f"name='{model_name}'")

    if not registered_versions:
        print(f"No registered versions found for '{model_name}'.")
        return

    print(f"\n==========================================")
    print(f"Model Registry Lifecycle Evaluation for '{model_name}'")
    print(f"Found {len(registered_versions)} registered version(s).")
    print(f"Promotion Rule: {promo_cfg['target_metric']} <= {promo_cfg['acceptance_threshold']}")
    print(f"==========================================")

    # Fetch metric for each version from its run
    version_metrics = []
    for mv in registered_versions:
        run = client.get_run(mv.run_id)
        rmse = run.data.metrics.get("rmse", 999.0)
        r2 = run.data.metrics.get("r2_score", -999.0)
        variant = run.data.tags.get("model_variant", "unknown")
        version_metrics.append({
            "version": int(mv.version),
            "run_id": mv.run_id,
            "rmse": rmse,
            "r2": r2,
            "variant": variant,
        })

    # Sort versions by performance (lowest RMSE is best)
    version_metrics.sort(key=lambda x: x["rmse"])
    best_candidate = version_metrics[0]

    for item in version_metrics:
        v = str(item["version"])
        run_id = item["run_id"]
        rmse = item["rmse"]
        variant = item["variant"]

        if item == best_candidate and rmse <= promo_cfg["acceptance_threshold"]:
            action = "Promoting to Production"
            decision_desc = f"Accepted: RMSE {rmse:.4f} meets threshold <= {promo_cfg['acceptance_threshold']}. Best candidate variant {variant}."
            # Set alias (modern MLflow) and stage (classic MLflow)
            try:
                client.set_registered_model_alias(model_name, "production", v)
            except Exception as e:
                print(f"Note on setting alias: {e}")
            try:
                client.transition_model_version_stage(
                    name=model_name, version=v, stage=promo_cfg["production_stage"], archive_existing_versions=True
                )
            except Exception as e:
                print(f"Note on stage transition: {e}")

            client.update_model_version(name=model_name, version=v, description=decision_desc)
            client.set_model_version_tag(name=model_name, version=v, key="lifecycle_status", value="Production")
            print(f" Version {v} ({variant}): {action} | RMSE={rmse:.4f} | Status=Production")

        elif rmse <= promo_cfg["acceptance_threshold"] * 1.2:
            action = "Marking as Candidate (Staging)"
            desc = f"Candidate for validation: RMSE {rmse:.4f}. Variant {variant}."
            try:
                client.set_registered_model_alias(model_name, "candidate", v)
            except Exception:
                pass
            try:
                client.transition_model_version_stage(name=model_name, version=v, stage=promo_cfg["candidate_stage"])
            except Exception:
                pass
            client.update_model_version(name=model_name, version=v, description=desc)
            client.set_model_version_tag(name=model_name, version=v, key="lifecycle_status", value="Staging")
            print(f" Version {v} ({variant}): {action} | RMSE={rmse:.4f} | Status=Staging")

        else:
            action = "Marking as Retired / Archived"
            desc = f"Archived/Baseline: RMSE {rmse:.4f}. Variant {variant} superseded."
            try:
                client.set_registered_model_alias(model_name, "archived", v)
            except Exception:
                pass
            try:
                client.transition_model_version_stage(name=model_name, version=v, stage=promo_cfg["retired_stage"])
            except Exception:
                pass
            client.update_model_version(name=model_name, version=v, description=desc)
            client.set_model_version_tag(name=model_name, version=v, key="lifecycle_status", value="Archived")
            print(f" Version {v} ({variant}): {action} | RMSE={rmse:.4f} | Status=Archived")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register MLflow models")
    parser.add_argument("--candidate", choices=["v1_ridge", "v2_random_forest", "v3_gradient_boosting", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.candidate == "all":
        for cand in ["v1_ridge", "v2_random_forest", "v3_gradient_boosting"]:
            train_candidate(cand, config, register=True)
        manage_model_lifecycle(config)
    else:
        train_candidate(args.candidate, config, register=True)
