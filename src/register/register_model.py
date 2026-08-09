"""Register a completed training job's model into the workspace model registry.

Usage:
    python -m src.register.register_model --job_name <job_name_printed_by_submit_training>

`job_name` must be the job that actually ran train.py (submit_training.py
submits a plain `command()` job, not a pipeline, so the name it prints is
already the right one -- no parent/child job resolution needed).
"""
from __future__ import annotations

import argparse

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model

from src.common.ml_client import get_ml_client


def register_model(job_name: str, model_name: str, description: str = "") -> Model:
    ml_client = get_ml_client()

    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/",
        name=model_name,
        description=description or f"Registered from job {job_name}",
        type=AssetTypes.MLFLOW_MODEL,
    )
    registered = ml_client.models.create_or_update(model)
    print(f"Registered model '{registered.name}' version {registered.version}")
    return registered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="breast_cancer_detection_model")  # TODO: rename if desired
    parser.add_argument("--description", type=str, default="")
    args = parser.parse_args()

    register_model(args.job_name, args.model_name, args.description)


if __name__ == "__main__":
    main()
