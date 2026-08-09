"""Single source of truth for connecting to the Azure ML workspace.

Every other script (notebook, training submitter, register, deploy) should
call `get_ml_client()` rather than building its own `MLClient`, so the
workspace connection details only need to be correct in one place (`.env`).
"""
from __future__ import annotations

import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


def get_ml_client() -> MLClient:
    """Build an authenticated MLClient for the workspace configured in `.env`.

    Requires a `.env` file (copy `.env.example`) with:
        AML_SUBSCRIPTION_ID
        AML_RESOURCE_GROUP
        AML_WORKSPACE_NAME

    Uses DefaultAzureCredential, which picks up (in order) an active
    `az login` session, environment-variable service principal creds,
    managed identity, etc. -- see:
    https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential
    """
    load_dotenv()

    subscription_id = os.environ.get("AML_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AML_RESOURCE_GROUP")
    workspace_name = os.environ.get("AML_WORKSPACE_NAME")

    missing = [
        name
        for name, value in [
            ("AML_SUBSCRIPTION_ID", subscription_id),
            ("AML_RESOURCE_GROUP", resource_group),
            ("AML_WORKSPACE_NAME", workspace_name),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required env var(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your workspace details."
        )

    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
