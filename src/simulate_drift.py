"""
simulate_drift.py
------------------
Task 3: Simulate "production" data that has drifted away from the
reference distribution, then score it with the saved model and write
the result to data/current.csv.

Drift is introduced in four ways, combined together, to make the
resulting drift report as illustrative as possible:
  1. Gaussian noise added to several numeric features (covariate/feature drift)
  2. A systematic shift (mean shift) applied to a couple of key features
  3. 10% of the true labels flipped (label/target drift & concept drift)
  4. One "important" feature dropped and back-filled with a constant so
     the model effectively loses access to it (feature removal drift)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 123  # different seed than train.py so "current" looks like a new production batch
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Features to inject Gaussian noise into
NOISY_FEATURES = ["mean radius", "mean texture", "mean smoothness", "worst concavity"]

# Features to apply a mean/systematic shift to (simulates sensor drift,
# population change, etc.)
SHIFTED_FEATURES = {
    "mean area": 1.25,       # multiply by 25%
    "worst perimeter": 1.15,  # multiply by 15%
}

# "Important" feature to effectively remove from production data
FEATURE_TO_REMOVE = "worst radius"

LABEL_FLIP_FRACTION = 0.10


def load_current_batch():
    """
    Simulate a new incoming batch of production data using a fresh,
    non-overlapping split of the same underlying dataset (a different
    random_state from train.py ensures a different set of rows).
    """
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    # Take a fresh 30% slice to act as the "new" production batch
    _, X_current, _, y_current = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    return X_current.reset_index(drop=True), y_current.reset_index(drop=True)


def inject_gaussian_noise(df, columns, noise_scale=0.5):
    df = df.copy()
    for col in columns:
        std = df[col].std()
        noise = np.random.normal(loc=0, scale=noise_scale * std, size=len(df))
        df[col] = df[col] + noise
    return df


def apply_feature_shift(df, shift_map):
    df = df.copy()
    for col, multiplier in shift_map.items():
        df[col] = df[col] * multiplier
    return df


def flip_labels(y, fraction=0.10, random_state=RANDOM_STATE):
    y_flipped = y.copy()
    rng = np.random.RandomState(random_state)
    n_flip = int(len(y_flipped) * fraction)
    flip_idx = rng.choice(y_flipped.index, size=n_flip, replace=False)
    y_flipped.loc[flip_idx] = 1 - y_flipped.loc[flip_idx]
    return y_flipped


def remove_important_feature(df, column):
    """Simulate an upstream pipeline failure where a feature stops being
    populated correctly and instead arrives as a constant/default value."""
    df = df.copy()
    df[column] = df[column].median()
    return df


def main():
    print("Loading model...")
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Simulating new production batch...")
    X_current, y_current = load_current_batch()

    print(f"Injecting Gaussian noise into: {NOISY_FEATURES}")
    X_current = inject_gaussian_noise(X_current, NOISY_FEATURES, noise_scale=0.6)

    print(f"Applying systematic shift to: {list(SHIFTED_FEATURES.keys())}")
    X_current = apply_feature_shift(X_current, SHIFTED_FEATURES)

    print(f"Removing important feature: '{FEATURE_TO_REMOVE}'")
    X_current = remove_important_feature(X_current, FEATURE_TO_REMOVE)

    print(f"Flipping {LABEL_FLIP_FRACTION * 100:.0f}% of labels...")
    y_current_flipped = flip_labels(y_current, fraction=LABEL_FLIP_FRACTION)

    print("Scoring drifted data with the trained model...")
    predictions = model.predict(X_current)

    current_df = X_current.copy()
    current_df["target"] = y_current_flipped.values
    current_df["prediction"] = predictions

    current_path = DATA_DIR / "current.csv"
    current_df.to_csv(current_path, index=False)
    print(f"Saved drifted 'current' dataset to {current_path} "
          f"({current_df.shape[0]} rows, {current_df.shape[1]} columns)")


if __name__ == "__main__":
    main()
