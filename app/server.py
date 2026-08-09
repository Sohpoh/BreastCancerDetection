"""Local FastAPI frontend for the deployed breast cancer detection endpoint.

Run:
    python app/server.py

Then open http://127.0.0.1:8000 in your browser.

Why this has to be a local server rather than a hosted static page: this
proxies requests to the real Azure ML endpoint using a bearer key
server-side. A browser calling the endpoint directly runs into two problems
-- managed online endpoints don't send CORS headers, so a fetch() straight
from any web page (even a local file) gets blocked by the browser, and
putting the auth key in client-side JS would expose it to anyone who views
the page source. This server sits in between: the browser only ever talks to
localhost, and this process is the only thing that holds the key and makes
the real (server-to-server, no CORS involved) call to Azure.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.common.ml_client import get_ml_client

ENDPOINT_NAME = "mlw-cancer-useast2-glyrj"  # TODO: change if you deploy a different endpoint

_scoring_uri: str | None = None
_auth_key: str | None = None


def _load_endpoint_credentials() -> None:
    global _scoring_uri, _auth_key
    ml_client = get_ml_client()
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    _scoring_uri = endpoint.scoring_uri
    _auth_key = ml_client.online_endpoints.get_keys(ENDPOINT_NAME).primary_key
    print(f"Loaded endpoint '{ENDPOINT_NAME}': {_scoring_uri}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_endpoint_credentials()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/predict")
def predict(record: dict[str, float]) -> dict:
    if _scoring_uri is None:
        raise HTTPException(status_code=500, detail="Endpoint credentials not loaded -- restart the server.")

    try:
        response = requests.post(
            _scoring_uri,
            headers={"Authorization": f"Bearer {_auth_key}", "Content-Type": "application/json"},
            json={"data": [record]},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()[0]
    except requests.exceptions.RequestException as e:
        detail = e.response.text if getattr(e, "response", None) is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Request to the endpoint failed: {detail}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
