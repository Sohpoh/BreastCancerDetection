"""Submit the training job to Azure ML.

Usage:
    python scripts/submit_training.py

Prints the AML studio URL and the job name -- pass that job name to
`src/register/register_model.py --job_name <name>` once you've reviewed the
run's metrics in the studio and decided it's worth registering. Training and
registration are deliberately separate steps (see `src/train/train.py`'s
docstring).
"""
from __future__ import annotations

from azure.ai.ml import Input, command

from src.common.ml_client import get_ml_client


def build_job():
    return command(
        inputs=dict(
            data=Input(
                type="uri_file",
                path="azureml:breast_cancer_dataset:1",
                mode="download",
            ),
            test_train_ratio=0.2,
            C=1.0,
            max_iter=100,
        ),
        code="./src/",  # location of source code
        command=(
            "python train/train.py "
            "--data ${{inputs.data}} "
            "--test_train_ratio ${{inputs.test_train_ratio}} "
            "--C ${{inputs.C}} "
            "--max_iter ${{inputs.max_iter}}"
        ),
        compute="spohane11",
        environment="aml-cancer-detect@latest",
        display_name="breast_cancer_prediction",
    )


def main() -> None:
    ml_client = get_ml_client()
    job = ml_client.jobs.create_or_update(build_job())
    print(f"Submitted job: {job.name}")
    print(f"Studio URL: {job.studio_url}")
    print(f"\nOnce complete, register with:")
    print(f"  python -m src.register.register_model --job_name {job.name}")


if __name__ == "__main__":
    main()
