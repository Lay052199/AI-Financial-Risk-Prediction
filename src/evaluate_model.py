from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report

from utils import (
    CLASSIFICATION_REPORT_PATH,
    EVAL_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    load_dataset,
    load_model,
    save_text_report,
)


def evaluate_saved_model() -> pd.DataFrame:
    df = load_dataset()
    model = load_model(MODEL_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["default"]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    evaluation_df = pd.DataFrame(
        {
            "actual": y,
            "predicted": predictions,
            "probability": probabilities,
        }
    )
    report = classification_report(y, predictions, digits=4)
    save_text_report(report, CLASSIFICATION_REPORT_PATH)
    return evaluation_df


if __name__ == "__main__":
    try:
        eval_df = evaluate_saved_model()
        print(f"Saved model evaluated on full dataset. Shape: {eval_df.shape}")
        if EVAL_PATH.exists():
            print(pd.read_csv(EVAL_PATH).to_string(index=False))
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
