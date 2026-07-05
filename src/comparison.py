from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = 42
DRIFT_SEED = 123

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

NOISY_FEATURES = ["mean radius", "mean texture", "mean smoothness", "worst concavity"]
SHIFTED_FEATURES = {"mean area": 1.25, "worst perimeter": 1.15}
FEATURE_TO_REMOVE = "worst radius"
LABEL_FLIP_FRACTION = 0.10


def load_split():
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_drifted_batch():
    """Recreate the same drifted 'current' batch used in simulate_drift.py."""
    np.random.seed(DRIFT_SEED)
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    _, X_current, _, y_current = train_test_split(
        X, y, test_size=0.30, random_state=DRIFT_SEED, stratify=y
    )
    X_current = X_current.reset_index(drop=True)
    y_current = y_current.reset_index(drop=True)

    for col in NOISY_FEATURES:
        std = X_current[col].std()
        X_current[col] = X_current[col] + np.random.normal(0, 0.6 * std, size=len(X_current))

    for col, mult in SHIFTED_FEATURES.items():
        X_current[col] = X_current[col] * mult

    X_current[FEATURE_TO_REMOVE] = X_current[FEATURE_TO_REMOVE].median()

    rng = np.random.RandomState(DRIFT_SEED)
    y_flipped = y_current.copy()
    n_flip = int(len(y_flipped) * LABEL_FLIP_FRACTION)
    flip_idx = rng.choice(y_flipped.index, size=n_flip, replace=False)
    y_flipped.loc[flip_idx] = 1 - y_flipped.loc[flip_idx]

    return X_current, y_flipped


def score(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
    }


def main():
    print("Loading data and training both models...")
    X_train, X_test, y_train, y_test = load_split()

    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    import pickle
    with open(MODEL_DIR / "model_rf.pkl", "wb") as f:
        pickle.dump(rf, f)

    print("Building the same drifted batch used in simulate_drift.py...")
    X_current, y_current = build_drifted_batch()

    results = {
        "Logistic Regression": {
            "reference": score(lr, X_test, y_test),
            "current": score(lr, X_current, y_current),
        },
        "Random Forest": {
            "reference": score(rf, X_test, y_test),
            "current": score(rf, X_current, y_current),
        },
    }

    print("\n=== Performance comparison: reference vs. drifted (current) data ===")
    for model_name, splits in results.items():
        print(f"\n{model_name}")
        for split_name, metrics in splits.items():
            metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            print(f"  {split_name:10s} -> {metric_str}")

    # --- Build grouped bar chart ---
    metrics_order = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics_order))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = [
        ("LR - Reference", results["Logistic Regression"]["reference"], "#4C72B0"),
        ("LR - Current (drifted)", results["Logistic Regression"]["current"], "#A6C8FF"),
        ("RF - Reference", results["Random Forest"]["reference"], "#C44E52"),
        ("RF - Current (drifted)", results["Random Forest"]["current"], "#F4A3A6"),
    ]

    for i, (label, metrics, color) in enumerate(bars):
        values = [metrics[m] for m in metrics_order]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Logistic Regression vs. Random Forest: Reference vs. Drifted Data")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "performance_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved performance comparison chart to {out_path}")


if __name__ == "__main__":
    main()
