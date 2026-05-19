from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_generator import generate_financial_risk_data
from utils import (
    CLASSIFICATION_REPORT_PATH,
    DATA_PATH,
    EVAL_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    create_distribution_plots,
    create_feature_importance_plot,
    create_shap_summary,
    ensure_directories,
    get_feature_importance,
    load_dataset,
    plot_confusion_matrix,
    plot_roc_curve,
    save_text_report,
)


RANDOM_STATE = 42


def build_models() -> Dict[str, Pipeline]:
    numeric_features = FEATURE_COLUMNS
    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    plain_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        ]
    )

    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", plain_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=8,
                        min_samples_split=8,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", plain_preprocessor),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", plain_preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    except ImportError:
        print("XGBoost is not installed. Skip XGBoost model.")

    return models


def evaluate_single_model(model: Pipeline, X_train, X_test, y_train, y_test) -> Tuple[dict, np.ndarray]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_prob


def train_models(df: pd.DataFrame) -> None:
    ensure_directories()
    create_distribution_plots(df)

    X = df[FEATURE_COLUMNS]
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()
    all_results = []
    trained_models: Dict[str, Pipeline] = {}
    probability_store: Dict[str, np.ndarray] = {}

    for model_name, pipeline in models.items():
        metrics, y_prob = evaluate_single_model(pipeline, X_train, X_test, y_train, y_test)
        trained_models[model_name] = pipeline
        probability_store[model_name] = y_prob
        all_results.append({"model": model_name, **metrics})
        print(f"{model_name}: {metrics}")

    results_df = pd.DataFrame(all_results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    results_df.to_csv(EVAL_PATH, index=False, encoding="utf-8")

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_prob = probability_store[best_model_name]
    y_pred = best_model.predict(X_test)

    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}: {best_model_name}")

    classification_text = classification_report(y_test, y_pred, digits=4)
    save_text_report(
        f"Best Model: {best_model_name}\n\n{classification_text}",
        CLASSIFICATION_REPORT_PATH,
    )

    plot_roc_curve(best_model_name, y_test, best_prob)
    plot_confusion_matrix(best_model, X_test, y_test)

    importance_df = get_feature_importance(best_model, FEATURE_COLUMNS)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8")
    create_feature_importance_plot(importance_df)

    X_train_processed = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    create_shap_summary(best_model, X_train_processed.sample(min(500, len(X_train_processed)), random_state=RANDOM_STATE))

    summary_lines = [
        f"Best Model: {best_model_name}",
        "",
        "Top Feature Importance:",
        importance_df.head(10).to_string(index=False),
        "",
        "Evaluation Summary:",
        results_df.to_string(index=False),
    ]
    save_text_report("\n".join(summary_lines), REPORTS_DIR / "training_summary.txt")


def main() -> None:
    if not DATA_PATH.exists():
        print("Dataset not found. Generating synthetic financial risk data first.")
        generate_financial_risk_data()

    try:
        df = load_dataset(DATA_PATH)
        train_models(df)
        print("Training pipeline completed successfully.")
    except Exception as exc:
        raise RuntimeError(f"Model training failed: {exc}") from exc


if __name__ == "__main__":
    main()
