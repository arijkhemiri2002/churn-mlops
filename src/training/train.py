"""Train several models and track every run with MLflow.

Usage (from the project root):
    python -m src.training.train
"""
from __future__ import annotations

import subprocess

import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocess import (
    build_preprocessor,
    clean_data,
    load_data,
    split_data,
)

CONFIG_PATH = "configs/train.yaml"

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_one(name, params, cfg, splits) -> dict:
    X_train, X_test, y_train, y_test = splits
    seed = cfg["seed"]

    model_cls = MODEL_CLASSES[name]
    model_params = dict(params)
    model_params["random_state"] = seed
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            ("model", model_cls(**model_params)),
        ]
    )

    with mlflow.start_run(run_name=name):
        # Reproducibility info: code version + data version
        mlflow.set_tags({"model_type": name, "git_commit": git_commit()})
        mlflow.log_artifact(cfg["data"]["dvc_file"], artifact_path="data_version")

        mlflow.log_params(model_params)
        mlflow.log_params(
            {"test_size": cfg["data"]["test_size"], "n_train": len(X_train)}
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        passed = metrics["recall"] >= cfg["quality_gate"]["min_recall"]
        mlflow.set_tag("quality_gate_passed", str(passed))

        # The whole pipeline (preprocessing + model) is logged, so the API
        # later receives raw features and does not need to re-implement them.
        signature = infer_signature(X_test, pipeline.predict(X_test))
        # Recent MLflow versions default to the "skops" format, which rejects some
        # numpy/xgboost types as untrusted. cloudpickle works for all 3 models.
        # (Security trade-off: only load models from sources you trust. Chapter 13.)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            serialization_format="cloudpickle",
        )

    return {"name": name, "passed": passed, **metrics}


def main() -> None:
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = clean_data(load_data(cfg["data"]["raw_path"]))
    splits = split_data(df, cfg["data"]["test_size"], cfg["seed"])

    results = [
        train_one(name, params, cfg, splits) for name, params in cfg["models"].items()
    ]

    print("\n=== Results (sorted by recall) ===")
    print(f"{'model':<22}{'recall':>8}{'precision':>11}{'f1':>8}{'roc_auc':>9}  gate")
    for r in sorted(results, key=lambda r: r["recall"], reverse=True):
        print(
            f"{r['name']:<22}{r['recall']:>8.3f}{r['precision']:>11.3f}"
            f"{r['f1']:>8.3f}{r['roc_auc']:>9.3f}  {'PASS' if r['passed'] else 'FAIL'}"
        )


if __name__ == "__main__":
    main()