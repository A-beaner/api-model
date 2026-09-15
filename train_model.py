"""
train_model.py - Train a simple ML model (RandomForest on Iris) and evaluate it.

API keys are read from environment variables - NEVER hardcode secrets in code.
Copy .env.example to .env and fill in your key, or export it:
    export API_KEY="your-real-key"
"""

import os

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


def main():
    # --- API key check (masked, never printed) ---
    api_key = os.environ.get("API_KEY")
    if api_key:
        print(f"API_KEY loaded from environment (***{api_key[-4:] if len(api_key) >= 4 else 'set'}***)")
    else:
        print("No API_KEY set - running in local/demo mode. Set it via env var or .env file.")

    # --- Load data ---
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features | train={len(X_train)}, test={len(X_test)}")

    # --- Train ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained: RandomForestClassifier(n_estimators=100)")

    # --- Evaluate ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=load_iris().target_names))

    # --- Save ---
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")


if __name__ == "__main__":
    main()
