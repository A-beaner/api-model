# api-model

An ML training pipeline powered by an NVIDIA API key:

1. Trains a RandomForest classifier on the Iris dataset
2. Evaluates it (accuracy + classification report) and saves `model.joblib`
3. Sends the results to an NVIDIA-hosted LLM (`openai/gpt-oss-20b` via
   `integrate.api.nvidia.com`) which writes a natural-language analysis

## API key setup (important)

Your NVIDIA API key is **never** stored in this repo. It's loaded from the
`API_KEY` environment variable:

```bash
cp .env.example .env   # then paste your real key into .env (gitignored)
# or:
export API_KEY="nvapi-..."
```

For CI, the key is stored as an encrypted GitHub Actions secret named
`API_KEY` (Settings -> Secrets and variables -> Actions). The workflow
injects it automatically - the key never appears in code or logs.

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
```

## CI

Every push triggers `.github/workflows/run.yml`, which installs dependencies
and runs the full pipeline (training + LLM analysis) using the `API_KEY`
secret. Latest verified result: **0.90 test accuracy** on the Iris hold-out
set, with LLM analysis generated via the NVIDIA API.
