from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils import (
    DATA_PATH,
    EVAL_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    FIELD_DESCRIPTIONS,
    FIGURES_DIR,
    MODEL_PATH,
    build_reason_text,
    load_dataset,
    load_model,
    probability_to_risk_level,
)


st.set_page_config(
    page_title="AI Financial Risk Prediction System",
    page_icon="💳",
    layout="wide",
)


def local_css() -> None:
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f6fbff 0%, #eef4f7 100%);
        }
        .hero-card {
            padding: 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #17324d 0%, #2a5f88 60%, #73a6c8 100%);
            color: white;
            box-shadow: 0 16px 40px rgba(23, 50, 77, 0.18);
        }
        .metric-card {
            padding: 1rem;
            border-radius: 14px;
            background: white;
            border: 1px solid rgba(23, 50, 77, 0.08);
            box-shadow: 0 8px 24px rgba(23, 50, 77, 0.08);
        }
        .small-note {
            color: #4f6475;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_project_intro() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>AI Financial Risk Prediction System</h1>
            <p>Machine Learning + Credit Risk Prediction + Explainable AI + Streamlit Dashboard</p>
            <p>Use synthetic lending data to analyze applicant risk, compare models, and deploy an interactive risk prediction application.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Project Background")
        st.write(
            "Financial institutions rely on data and AI models to identify high-risk customers before loan approval. "
            "This project demonstrates a full credit risk workflow using synthetic data for portfolio-style presentation."
        )
        st.subheader("Business Value")
        st.write(
            "It can support credit review, risk management, customer segmentation, and lending decision analysis."
        )
    with col2:
        st.subheader("Technical Pipeline")
        st.write(
            "Data Generation -> EDA -> Feature Analysis -> Machine Learning Modeling -> Model Evaluation -> Risk Prediction -> Explainable AI"
        )
        st.subheader("Project Highlights")
        st.write(
            "- Multi-model comparison\n"
            "- Default probability prediction\n"
            "- Explainable AI analysis\n"
            "- Streamlit visualization\n"
            "- GitHub portfolio readiness"
        )


def show_dataset_page() -> None:
    st.header("Data Analysis")
    if not DATA_PATH.exists():
        st.warning("Dataset not found. Run `python src/data_generator.py` first.")
        return

    df = load_dataset(DATA_PATH)
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", f"{len(df):,}")
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Default Rate", f"{df['default'].mean():.2%}")

    st.subheader("Top 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Field Description")
    field_df = pd.DataFrame(
        {"field": list(FIELD_DESCRIPTIONS.keys()), "description": list(FIELD_DESCRIPTIONS.values())}
    )
    st.dataframe(field_df, use_container_width=True)

    figure_files = [
        "default_distribution.png",
        "credit_score_distribution.png",
        "income_distribution.png",
        "loan_amount_distribution.png",
        "dti_distribution.png",
        "correlation_heatmap.png",
    ]
    st.subheader("Saved Charts")
    cols = st.columns(2)
    for index, filename in enumerate(figure_files):
        figure_path = FIGURES_DIR / filename
        if figure_path.exists():
            cols[index % 2].image(str(figure_path), caption=filename.replace(".png", "").replace("_", " ").title())


from pathlib import Path

@st.cache_resource
def load_prediction_model():
    """加载金融风险预测模型。"""
    if not MODEL_PATH.exists():
        st.error(f"模型文件不存在：{MODEL_PATH}")
        return None

    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("模型加载失败，请检查 best_model.pkl 是否与当前 Python / scikit-learn 版本兼容。")
        st.exception(e)
        return None

def build_input_form() -> pd.DataFrame:
    st.sidebar.header("Customer Input")
    age = st.sidebar.slider("Age", 21, 65, 35)
    income = st.sidebar.number_input("Annual Income", min_value=50000, max_value=1000000, value=180000, step=10000)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=10000, max_value=800000, value=150000, step=10000)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    employment_years = st.sidebar.slider("Employment Years", 0, 40, 6)
    debt_to_income_ratio = st.sidebar.slider("Debt to Income Ratio", 0.05, 0.95, 0.32, 0.01)
    overdue_times = st.sidebar.slider("Overdue Times", 0, 8, 1)
    loan_term = st.sidebar.selectbox("Loan Term", [12, 24, 36, 48, 60], index=2)
    has_house = st.sidebar.selectbox("Has House", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    has_car = st.sidebar.selectbox("Has Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    return pd.DataFrame(
        [
            {
                "age": age,
                "income": income,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "employment_years": employment_years,
                "debt_to_income_ratio": debt_to_income_ratio,
                "overdue_times": overdue_times,
                "loan_term": loan_term,
                "has_house": has_house,
                "has_car": has_car,
            }
        ]
    )


def show_prediction_page():
    st.title("Risk Prediction")

    model = load_prediction_model()

    if model is None:
        st.warning("当前模型未成功加载，暂时无法进行风险预测。请检查 models/best_model.pkl 文件和部署环境。")
        return

    st.markdown("### Please enter the financial indicators below:")

    input_df = build_input_form()
    st.subheader("Applicant Snapshot")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Risk", type="primary"):
        probability = float(model.predict_proba(input_df[FEATURE_COLUMNS])[:, 1][0])
        prediction = int(model.predict(input_df[FEATURE_COLUMNS])[0])
        risk_level, note = probability_to_risk_level(probability)
        label = "High Risk / Default" if prediction == 1 else "Low Risk / Non-default"

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", label)
        col2.metric("Default Probability", f"{probability:.2%}")
        col3.metric("Risk Level", risk_level)

        st.info(note)
        st.write(build_reason_text(input_df))


def show_explainability_page() -> None:
    st.header("Model Explainability")
    if not EVAL_PATH.exists() or not FEATURE_IMPORTANCE_PATH.exists():
        st.warning("Evaluation outputs not found. Run `python src/train_model.py` first.")
        return

    eval_df = pd.read_csv(EVAL_PATH)
    st.subheader("Best Model and Metrics")
    st.dataframe(eval_df, use_container_width=True)
    st.write(f"Best model by ROC-AUC: **{eval_df.iloc[0]['model']}**")

    st.subheader("Feature Importance")
    feature_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    st.dataframe(feature_df, use_container_width=True)

    feature_plot = FIGURES_DIR / "feature_importance.png"
    if feature_plot.exists():
        st.image(str(feature_plot), caption="Feature Importance")

    roc_plot = FIGURES_DIR / "roc_curve.png"
    cm_plot = FIGURES_DIR / "confusion_matrix.png"
    cols = st.columns(2)
    if roc_plot.exists():
        cols[0].image(str(roc_plot), caption="ROC Curve")
    if cm_plot.exists():
        cols[1].image(str(cm_plot), caption="Confusion Matrix")

    shap_plot = FIGURES_DIR / "shap_summary.png"
    if shap_plot.exists():
        st.subheader("SHAP Summary")
        st.image(str(shap_plot), caption="SHAP Summary Plot")

    st.subheader("Business Interpretation")
    st.write(
        "Historical overdue behavior, credit score, debt burden, and loan exposure are usually the most important signals "
        "in credit risk assessment. Lower credit quality, more previous delinquencies, higher debt-to-income ratio, and "
        "larger requested loan amounts generally correspond to higher default risk."
    )
    st.write(
        "Income, employment stability, house ownership, and car ownership act as supplementary stability indicators. "
        "These variables help the model approximate repayment capacity and asset resilience."
    )


def main() -> None:
    local_css()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Project Introduction", "Data Analysis", "Risk Prediction", "Model Explainability"],
    )

    if page == "Project Introduction":
        show_project_intro()
    elif page == "Data Analysis":
        show_dataset_page()
    elif page == "Risk Prediction":
        show_prediction_page()
    else:
        show_explainability_page()

    st.caption(
        "This project uses synthetic data for learning, portfolio presentation, and interview discussion only."
    )


if __name__ == "__main__":
    main()
