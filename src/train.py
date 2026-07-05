
import pickle
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data():
    """Load the Breast Cancer Wisconsin dataset as a DataFrame."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target  # 0 = malignant, 1 = benign
    return X, y


def train_model(X_train, y_train):
    """Train a simple Logistic Regression classifier."""
    model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    print(f"[{label}] Accuracy={acc:.4f} Precision={prec:.4f} "
          f"Recall={rec:.4f} F1={f1:.4f}")
    return preds


def main():
    print("Loading Breast Cancer Wisconsin dataset...")
    X, y = load_data()

    # Split: the "reference" split represents the trusted/production
    # baseline data the model was trained and validated on.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print("Training Logistic Regression model...")
    model = train_model(X_train, y_train)

    evaluate(model, X_train, y_train, label="train")
    evaluate(model, X_test, y_test, label="reference (test split)")

    # Save the trained model
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to {model_path}")

    # Build reference.csv: held-out test data + true target + model predictions
    reference_df = X_test.copy()
    reference_df["target"] = y_test.values
    reference_df["prediction"] = model.predict(X_test)

    reference_path = DATA_DIR / "reference.csv"
    reference_df.to_csv(reference_path, index=False)
    print(f"Saved reference dataset to {reference_path} "
          f"({reference_df.shape[0]} rows, {reference_df.shape[1]} columns)")


if __name__ == "__main__":
    main()
