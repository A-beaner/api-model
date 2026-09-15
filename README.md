# api-model

A small ML training pipeline. Trains a RandomForest classifier on the Iris
dataset, evaluates it, and saves the model to `model.joblib`.

## API key setup (important)

Your API key is **never** stored in this repo. It is loaded from an
environment variable:

```bash
cp .env.example .env   # then paste your real key into .env (gitignored)
# or:
export API_KEY="your-real-key"
```

For CI, add it as a GitHub Actions secret:
**Settings -> Secrets and variables -> Actions -> New repository secret -> `API_KEY`**

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
```

## CI

Every push triggers `.github/workflows/run.yml`, which installs dependencies
and runs the training script. Latest verified result: **0.90 test accuracy**
on the Iris hold-out set.
