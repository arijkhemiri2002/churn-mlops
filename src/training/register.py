"""Model Registry: register the best run, then promote it with aliases.

Usage (from the project root):
    python -m src.training.register register   # best run -> new version, alias "staging"
    python -m src.training.register promote    # staging -> production (gate re-checked)
    python -m src.training.register verify     # load the production model and predict

Selection rule (same as in the README):
    highest recall among finished runs that passed the quality gate,
    with F1 as the tie-breaker.
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.features.preprocess import TARGET, clean_data, load_data
from src.training.train import load_config

MODEL_NAME = "churn-classifier"


def find_best_run(cfg: dict):
    runs = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["experiment_name"]],
        filter_string=(
            "attributes.status = 'FINISHED' AND tags.quality_gate_passed = 'True'"
        ),
        order_by=["metrics.recall DESC", "metrics.f1 DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(
            "No finished run passed the quality gate. Train again or tune the models."
        )
    return runs.iloc[0]


def register(cfg: dict, client: MlflowClient) -> None:
    best = find_best_run(cfg)
    run_id = best["run_id"]
    recall, f1 = best["metrics.recall"], best["metrics.f1"]
    model_type = best.get("tags.model_type", "unknown")
    print(f"Best run: {model_type} (run {run_id[:8]}) recall={recall:.3f} f1={f1:.3f}")

    # Idempotent: do not create a second version for the same run
    already = [
        mv
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
        if mv.run_id == run_id
    ]
    if already:
        version = already[0].version
        print(f"This run is already registered as version {version}.")
    else:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        version = mv.version
        for key, value in {
            "recall": f"{recall:.4f}",
            "f1": f"{f1:.4f}",
            "model_type": model_type,
            "git_commit": best.get("tags.git_commit", "unknown"),
            "source_run_id": run_id,
        }.items():
            client.set_model_version_tag(MODEL_NAME, version, key, value)
        print(f"Registered {MODEL_NAME} version {version}.")

    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    print(f"Alias 'staging' -> version {version}")


def promote(cfg: dict, client: MlflowClient) -> None:
    try:
        staging = client.get_model_version_by_alias(MODEL_NAME, "staging")
    except MlflowException:
        raise SystemExit("No version has the alias 'staging'. Run 'register' first.")

    # Re-check the quality gate at promotion time, not only at training time
    recall = client.get_run(staging.run_id).data.metrics["recall"]
    min_recall = cfg["quality_gate"]["min_recall"]
    if recall < min_recall:
        raise SystemExit(
            f"Promotion blocked: recall {recall:.3f} < gate {min_recall:.2f}."
        )

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "production")
    except MlflowException:
        current = None

    if current is not None and current.version == staging.version:
        print(f"Version {staging.version} is already in production.")
        return

    if current is not None:
        # Keep the old version reachable for a quick rollback
        client.set_registered_model_alias(MODEL_NAME, "previous", current.version)
        print(f"Alias 'previous' -> version {current.version} (rollback)")

    client.set_registered_model_alias(MODEL_NAME, "production", staging.version)
    print(f"Alias 'production' -> version {staging.version} (recall={recall:.3f})")


def verify(cfg: dict) -> None:
    uri = f"models:/{MODEL_NAME}@production"
    model = mlflow.sklearn.load_model(uri)
    df = clean_data(load_data(cfg["data"]["raw_path"]))
    X = df.drop(columns=[TARGET]).head(5)
    proba = model.predict_proba(X)[:, 1]
    print(f"Loaded {uri}")
    print("Churn probabilities for 5 customers:", proba.round(3))
    print("Actual labels:                      ", df[TARGET].head(5).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["register", "promote", "verify"])
    action = parser.parse_args().action

    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()

    if action == "register":
        register(cfg, client)
    elif action == "promote":
        promote(cfg, client)
    else:
        verify(cfg)


if __name__ == "__main__":
    main()