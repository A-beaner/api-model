"""
train_model.py - Train an ML model (RandomForest on Iris), evaluate it, and use an
NVIDIA-hosted LLM (via your NVIDIA API key) to generate an analysis of the results.

The API key is read from the API_KEY environment variable - NEVER hardcode secrets.
Copy .env.example to .env and fill in your key, or: export API_KEY="your-key"
"""

import json
import os
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env if present (gitignored - secrets stay local)
except ImportError:
    pass

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"


def train_and_evaluate():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=load_iris().target_names)
    joblib.dump(model, "model.joblib")
    return acc, report


def llm_analysis(acc, report, api_key):
    """Ask an NVIDIA-hosted LLM to interpret the training results."""
    prompt = (
        "A RandomForest classifier was trained on the Iris dataset. "
        f"Test accuracy: {acc:.4f}.\nClassification report:\n{report}\n\n"
        "In 3-4 sentences, interpret these results for an ML engineer: "
        "is the model good, which classes are hardest, and one suggestion to improve."
    )
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        NVIDIA_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def main():
    api_key = os.environ.get("API_KEY")
    if api_key:
        masked = api_key[-4:] if len(api_key) >= 4 else "set"
        print(f"API_KEY loaded from environment (***{masked})")
    else:
        print("No API_KEY set - LLM analysis will be skipped.")

    print("Training RandomForestClassifier on Iris...")
    acc, report = train_and_evaluate()
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(report)
    print("Model saved to model.joblib")

    if api_key:
        print(f"\n--- LLM analysis ({LLM_MODEL} via NVIDIA API) ---")
        try:
            print(llm_analysis(acc, report, api_key))
        except Exception as e:
            print(f"LLM analysis failed (training results above are still valid): {e}")


if __name__ == "__main__":
    main()
