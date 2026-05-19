from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DATA_PATH = DATA_DIR / "financial_risk_data.csv"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
EVAL_PATH = REPORTS_DIR / "model_evaluation.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary.png"


FEATURE_COLUMNS: List[str] = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_to_income_ratio",
    "overdue_times",
    "loan_term",
    "has_house",
    "has_car",
]

FIELD_DESCRIPTIONS: Dict[str, str] = {
    "age": "Customer age",
    "income": "Annual income in RMB",
    "loan_amount": "Loan amount requested",
    "credit_score": "Credit score between 300 and 850",
    "employment_years": "Years of employment",
    "debt_to_income_ratio": "Debt-to-income ratio",
    "overdue_times": "Historical overdue count",
    "loan_term": "Loan term in months",
    "has_house": "Owns a house: 1 yes, 0 no",
    "has_car": "Owns a car: 1 yes, 0 no",
    "default": "Target label: 1 default/high risk, 0 non-default/low risk",
}


def ensure_directories() -> None:
    for directory in [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, FIGURES_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="Blues")
    plt.rcParams["figure.figsize"] = (9, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["figure.dpi"] = 120


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_directories()
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    ensure_directories()
    target_path = data_path or DATA_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {target_path}. Run `python src/data_generator.py` first."
        )
    return pd.read_csv(target_path)


def load_model(model_path: Path | None = None):
    target_path = model_path or MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model not found at {target_path}. Run `python src/train_model.py` first."
        )
    return joblib.load(target_path)


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    else:
        importance = np.zeros(len(feature_names))

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


def create_feature_importance_plot(importance_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#3f7aa3")
    ax.set_title("Feature Importance Ranking")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return save_figure(fig, "feature_importance.png")


def create_distribution_plots(df: pd.DataFrame) -> None:
    set_plot_style()

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="default", ax=ax, color="#4C78A8")
    ax.set_title("Default vs Non-default Distribution")
    ax.set_xlabel("Default")
    ax.set_ylabel("Count")
    ax.set_xticks([0, 1], labels=["Non-default", "Default"])
    save_figure(fig, "default_distribution.png")

    hist_configs = [
        ("credit_score", "Credit Score Distribution", "credit_score_distribution.png"),
        ("income", "Income Distribution", "income_distribution.png"),
        ("loan_amount", "Loan Amount Distribution", "loan_amount_distribution.png"),
        (
            "debt_to_income_ratio",
            "Debt-to-Income Ratio Distribution",
            "dti_distribution.png",
        ),
    ]
    for column, title, filename in hist_configs:
        fig, ax = plt.subplots()
        sns.histplot(df[column], bins=30, kde=True, ax=ax, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        save_figure(fig, filename)

    risk_relation_configs = [
        ("credit_score", "Credit Score vs Default Risk", "credit_score_vs_default.png"),
        (
            "debt_to_income_ratio",
            "Debt-to-Income Ratio vs Default Risk",
            "dti_vs_default.png",
        ),
        ("income", "Income vs Default Risk", "income_vs_default.png"),
    ]
    for column, title, filename in risk_relation_configs:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="default", y=column, ax=ax, color="#72B7B2")
        ax.set_title(title)
        ax.set_xlabel("Default")
        ax.set_ylabel(column)
        ax.set_xticks([0, 1], labels=["Non-default", "Default"])
        save_figure(fig, filename)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="Blues", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_figure(fig, "correlation_heatmap.png")


def plot_roc_curve(model_name: str, y_true, y_score) -> Path:
    set_plot_style()
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=model_name)
    ax.set_title("ROC Curve")
    return save_figure(fig, "roc_curve.png")


def plot_confusion_matrix(model, X_test, y_test) -> Path:
    set_plot_style()
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    return save_figure(fig, "confusion_matrix.png")


def save_text_report(text: str, output_path: Path) -> None:
    ensure_directories()
    output_path.write_text(text, encoding="utf-8")


def probability_to_risk_level(probability: float) -> Tuple[str, str]:
    if probability < 0.30:
        return "Low Risk", "The predicted default probability is low. Standard approval rules are generally acceptable."
    if probability < 0.60:
        return "Medium Risk", "The applicant shows some pressure signals. Consider tighter review and income verification."
    return "High Risk", "The applicant has a high expected default risk. Consider stricter approval, lower limits, or extra collateral."


def build_reason_text(input_df: pd.DataFrame) -> str:
    row = input_df.iloc[0]
    reasons: List[str] = []
    if row["overdue_times"] >= 3:
        reasons.append("multiple historical overdue records")
    if row["credit_score"] < 600:
        reasons.append("low credit score")
    if row["debt_to_income_ratio"] >= 0.45:
        reasons.append("high debt-to-income ratio")
    if row["income"] < 120000:
        reasons.append("relatively low income")
    if row["employment_years"] < 2:
        reasons.append("short employment history")
    if row["loan_amount"] > 350000:
        reasons.append("large loan exposure")
    if row["has_house"] == 0:
        reasons.append("no house ownership signal")
    if row["has_car"] == 0:
        reasons.append("no car ownership signal")

    if not reasons:
        return "The applicant shows relatively stable core indicators with limited obvious risk signals."
    return "Main risk signals: " + ", ".join(reasons) + "."


def create_shap_summary(model, X_train: pd.DataFrame) -> bool:
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Skip SHAP analysis.")
        return False

    estimator = model.named_steps.get("model", model)
    try:
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_train)
            values = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            explainer = shap.Explainer(model.predict, X_train)
            values = explainer(X_train)

        plt.figure()
        shap.summary_plot(values, X_train, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PATH, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary saved to {SHAP_SUMMARY_PATH}")
        return True
    except Exception as exc:  # pragma: no cover - SHAP environment dependent
        print(f"SHAP analysis failed and was skipped: {exc}")
        plt.close("all")
        return False
