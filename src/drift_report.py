"""
drift_report.py
----------------
Task 4: Use Evidently AI to compare the reference dataset against the
current (drifted) dataset and generate an HTML report covering:
  - Classification quality metrics (Accuracy, Precision, Recall, F1)
  - Feature (data) drift
  - Prediction drift
  - Distribution changes
  - Target drift
  - Visual charts

Output: reports/model_drift_report.html
"""

from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import BinaryClassification
from evidently.presets import DataDriftPreset, DataSummaryPreset, ClassificationPreset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    print("Loading reference and current datasets...")
    reference_df = pd.read_csv(DATA_DIR / "reference.csv")
    current_df = pd.read_csv(DATA_DIR / "current.csv")

    feature_columns = [c for c in reference_df.columns if c not in ("target", "prediction")]

    data_definition = DataDefinition(
        numerical_columns=feature_columns,
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
            )
        ],
    )

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    print("Running Evidently report (Data Drift + Data Summary + Classification Quality)...")
    report = Report(
        metrics=[
            DataDriftPreset(),      # feature drift, prediction drift, target drift, distribution changes
            DataSummaryPreset(),    # descriptive stats / distribution changes for each column
            ClassificationPreset(), # accuracy, precision, recall, f1 + confusion matrix
        ]
    )

    my_eval = report.run(current_data=current_dataset, reference_data=reference_dataset)

    report_path = REPORTS_DIR / "model_drift_report.html"
    my_eval.save_html(str(report_path))
    print(f"Saved Evidently drift report to {report_path}")

    # Also print a short console summary so it's visible without opening the HTML
    result_dict = my_eval.dict()
    print("\n--- Quick summary (see HTML report for full detail) ---")
    print(f"Report generated with {len(result_dict.get('metrics', []))} top-level metric blocks.")


if __name__ == "__main__":
    main()
