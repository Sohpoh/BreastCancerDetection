# Breast Cancer Detection — Azure ML project

Data analysis → training → registration → (deployment, coming later), using
the Azure ML Python SDK v2 against workspace `mlw-cancer-useast2`.

## Layout

```
config/environments/train-env.yml   Reference conda spec for the training environment
notebooks/01_data_exploration.ipynb  EDA against the breast_cancer_dataset data asset
src/common/ml_client.py              get_ml_client() -- shared workspace connection (reads .env)
src/train/train.py                   Trains a LogisticRegression, logs it as an MLflow run artifact
src/register/register_model.py       Registers a completed run's model into the model registry
scripts/submit_training.py           Submits the training job to Azure ML
tests/test_train.py                  Offline unit tests for the training logic (no AML calls)
```

## One-time setup

```bash
# Use environment.yml, or pip-install into an existing conda env of your own:
conda env create -f environment.yml && conda activate breast-cancer-detection
# --- or, in an existing env ---
pip install azure-ai-ml azure-identity mlflow azureml-mlflow python-dotenv

pip install -e .              # makes `src`/`scripts` importable from anywhere
cp .env.example .env           # fill in AML_SUBSCRIPTION_ID / AML_RESOURCE_GROUP / AML_WORKSPACE_NAME
az login                       # DefaultAzureCredential picks this up
```

`scripts/submit_training.py` currently targets:
- **Data asset**: `breast_cancer_dataset:1` (`uri_file` — a single Kaggle-format CSV)
- **Compute**: `spohane11`
- **Environment**: `aml-cancer-detect@latest` (must already be registered in the
  workspace; `config/environments/train-env.yml` is a reference spec if you
  ever need to (re)create it via `az ml environment create --file
  config/environments/train-env.yml --resource-group <rg> --workspace-name <workspace>`)

If any of those change, edit the constants at the top of `build_job()` in
`scripts/submit_training.py`.

## Workflow

1. **Explore**: run `notebooks/01_data_exploration.ipynb` (already pointed at
   `breast_cancer_dataset:1`) to sanity-check the data.
2. **Train**:
   ```bash
   python scripts/submit_training.py
   ```
   Submits a command job running `src/train/train.py` (LogisticRegression +
   StandardScaler on the `diagnosis` column, `id`/`Unnamed: 32` dropped
   automatically). Prints the job name, studio URL, and the exact
   `register_model.py` command to run next.
3. **Register**: after checking the run's metrics in the studio,
   ```bash
   python -m src.register.register_model --job_name <job_name_from_step_2>
   ```
   Registers as `breast_cancer_detection_model` by default (`--model_name` to
   override). Training and registration are deliberate separate steps so a
   bad run doesn't automatically get promoted.
4. **Deploy**: not yet implemented.

## Tests

```bash
python -m pytest tests/
```

Covers `load_data` (Kaggle boilerplate column dropping), `prepare_features`
(diagnosis encoding), and `train_and_evaluate` (fit + metrics) against
synthetic data — no AML/network calls.
