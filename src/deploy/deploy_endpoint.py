"""Create/update a managed online endpoint and deploy a registered model to it.

Usage:
    python -m src.deploy.deploy_endpoint --model_name breast_cancer_detection_model

An endpoint is a stable HTTPS URL + auth; a deployment is the actual model +
compute serving traffic behind it (an endpoint can hold multiple deployments,
e.g. for blue/green rollouts, with traffic split by percentage -- that's why
"create the endpoint", "create the deployment", and "route traffic to it" are
three separate calls below).

This is a *custom* deployment (not Azure's no-code MLflow path): it uses
src/deploy/score.py as the scoring entry script, running in the environment
defined by config/environments/score-env.yml. That's deliberate -- a no-code
deployment can only return the bare prediction, but score.py returns a
per-request explanation (which features drove *this* prediction) using the
fact that the model is a linear LogisticRegression, see score.py's docstring
for the math. If you don't need that, a no-code deployment (drop
code_configuration=/environment= below, just pass model=) is simpler.

Notes before running:
  - Endpoint names must be globally unique within the region (they become
    part of the DNS hostname) -- if creation fails with a name conflict,
    pick a different --endpoint_name.
  - Needs VM core quota for --instance_type in your subscription/region.
  - Deployment (step 2) is slow -- expect 10-15 min the first time (image
    build + VM provisioning).
  - This keeps a VM running (and billing) continuously once deployed --
    delete it when done: `az ml online-endpoint delete --name <name> --yes`.
"""
from __future__ import annotations

import argparse

from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)

from src.common.ml_client import get_ml_client

SCORE_ENV_CONDA_FILE = "./config/environments/score-env.yml"
# Matches the base image used by aml-cancer-detect (the training environment)
# for consistency, though the custom scoring conda file above is what
# actually pins the Python/library versions here.
SCORE_ENV_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"


def get_latest_model(ml_client, model_name: str):
    """Azure ML SDK v2 has no "latest" shortcut for models -- scan versions."""
    latest_version = max(int(m.version) for m in ml_client.models.list(name=model_name))
    return ml_client.models.get(name=model_name, version=str(latest_version))


def deploy(
    model_name: str,
    endpoint_name: str,
    instance_type: str = "Standard_DS2_v2",
    instance_count: int = 1,
) -> str:
    ml_client = get_ml_client()

    model = get_latest_model(ml_client, model_name)
    print(f"Deploying {model.name} v{model.version}")

    print(f"[1/3] Creating/updating endpoint '{endpoint_name}'...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Breast cancer detection -- LogisticRegression with per-prediction feature explanations.",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment_name = "default"
    print(f"[2/3] Creating/updating deployment '{deployment_name}' (this is slow -- 10-15 min first time)...")
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        code_configuration=CodeConfiguration(code="./src/deploy", scoring_script="score.py"),
        environment=Environment(conda_file=SCORE_ENV_CONDA_FILE, image=SCORE_ENV_BASE_IMAGE),
        instance_type=instance_type,
        instance_count=instance_count,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    print(f"[3/3] Routing 100% of traffic to '{deployment_name}'...")
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    scoring_uri = ml_client.online_endpoints.get(endpoint_name).scoring_uri
    print(f"Deployed. Scoring URI: {scoring_uri}")
    print(f"Get an auth key with: az ml online-endpoint get-credentials --name {endpoint_name}")
    return scoring_uri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--endpoint_name", type=str, default="breast-cancer-endpoint")  # TODO: rename if this name is taken
    parser.add_argument("--instance_type", type=str, default="Standard_DS2_v2")
    parser.add_argument("--instance_count", type=int, default=1)
    args = parser.parse_args()

    deploy(
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )


if __name__ == "__main__":
    main()
