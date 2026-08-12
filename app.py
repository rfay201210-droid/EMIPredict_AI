"""
EMIPredict AI - Intelligent Financial Risk Assessment Platform
Multi-page Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
DATA_PATH = BASE / "data" / "EMI_dataset.csv"
REPORTS_DIR = BASE / "reports"

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #1a5276, #2980b9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #5d6d7e; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .eligible { color: #27ae60; font-weight: 700; font-size: 1.4rem; }
    .highrisk { color: #f39c12; font-weight: 700; font-size: 1.4rem; }
    .noteligible { color: #e74c3c; font-weight: 700; font-size: 1.4rem; }
    .stButton>button {
        background: linear-gradient(90deg, #2980b9, #1a5276);
        color: white; border-radius: 8px; border: none;
        padding: 0.55rem 1.4rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        clf = joblib.load(MODELS_DIR / "best_classification_model.joblib")
        reg = joblib.load(MODELS_DIR / "best_regression_model.joblib")
        le = joblib.load(MODELS_DIR / "label_encoder_eligibility.joblib")
        meta = joblib.load(MODELS_DIR / "model_metadata.joblib")
        return clf, reg, le, meta
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return None, None, None, None


@st.cache_data
def load_sample_data(n=5000):
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False, nrows=n * 3)
        for c in ["age", "monthly_salary", "bank_balance"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sample(min(n, len(df)), random_state=42)
    except Exception:
        return None


def sidebar_nav():
    st.sidebar.markdown("## 💰 EMIPredict AI")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 EMI Prediction", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.info("**Dual ML Platform**\n- Classification: EMI Eligibility\n- Regression: Max Safe EMI")
    return page


def page_home():
    st.markdown('<p class="main-header">EMIPredict AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent Financial Risk Assessment Platform for EMI Decisions</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records Analyzed", "400,000+")
    c2.metric("Classification Acc", "94.3%")
    c3.metric("Regression R²", "0.97")
    c4.metric("EMI Scenarios", "5")
    st.markdown("---")
    st.subheader("🎯 What this platform does")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        **For Customers & Borrowers**
        - Instant EMI eligibility check
        - Personalized maximum safe EMI recommendation
        - Transparent risk scoring
        - Better financial planning before applying
        """
        )
    with col2:
        st.markdown(
            """
        **For Banks & FinTech**
        - Automate underwriting decisions
        - Risk-based pricing support
        - Reduce manual review time significantly
        - Consistent, data-driven policy
        """
        )
    st.subheader("📋 Supported EMI Scenarios")
    scenarios = pd.DataFrame(
        {
            "Scenario": [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI",
            ],
            "Typical Amount (INR)": ["10K – 200K", "20K – 300K", "80K – 1.5M", "50K – 1M", "50K – 500K"],
            "Tenure (months)": ["3 – 24", "6 – 36", "12 – 84", "12 – 60", "6 – 48"],
        }
    )
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    st.success("👉 Go to **EMI Prediction** page to try a live assessment.")


def build_input_form():
    st.subheader("Enter Applicant Details")
    with st.expander("👤 Personal & Employment", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 25, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        with c2:
            education = st.selectbox(
                "Education", ["High School", "Graduate", "Post Graduate", "Professional"]
            )
            employment_type = st.selectbox(
                "Employment Type", ["Private", "Government", "Self-employed"]
            )
            years_of_employment = st.slider("Years of Employment", 0.5, 30.0, 4.0, 0.5)
        with c3:
            company_type = st.selectbox(
                "Company Type", ["Startup", "Mid-size", "MNC", "Public Sector"]
            )
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
            monthly_rent = st.number_input("Monthly Rent (INR)", 0, 80000, 8000, 500)

    with st.expander("👨‍👩‍👧 Family & Dependents"):
        c1, c2, c3 = st.columns(3)
        with c1:
            family_size = st.slider("Family Size", 1, 5, 3)
            dependents = st.slider("Dependents", 0, 4, 1)
        with c2:
            school_fees = st.number_input("School Fees / month", 0, 15000, 0, 500)
            college_fees = st.number_input("College Fees / month", 0, 25000, 0, 500)
        with c3:
            travel_expenses = st.number_input("Travel Expenses", 600, 30000, 4000, 200)
            groceries_utilities = st.number_input(
                "Groceries & Utilities", 1800, 70000, 10000, 500
            )

    with st.expander("💵 Income & Credit"):
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_salary = st.number_input(
                "Monthly Salary (INR)", 15000, 200000, 55000, 1000
            )
            other_monthly_expenses = st.number_input(
                "Other Monthly Expenses", 600, 40000, 5000, 500
            )
        with c2:
            existing_loans = st.selectbox("Existing Loans?", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI Amount", 0, 50000, 0, 500)
        with c3:
            credit_score = st.slider("Credit Score", 300, 850, 720)
            bank_balance = st.number_input("Bank Balance", 0, 2000000, 80000, 5000)
            emergency_fund = st.number_input("Emergency Fund", 0, 500000, 30000, 2000)

    with st.expander("📝 Loan Request"):
        c1, c2, c3 = st.columns(3)
        with c1:
            emi_scenario = st.selectbox(
                "EMI Scenario",
                [
                    "E-commerce Shopping EMI",
                    "Home Appliances EMI",
                    "Vehicle EMI",
                    "Personal Loan EMI",
                    "Education EMI",
                ],
            )
        with c2:
            requested_amount = st.number_input(
                "Requested Amount (INR)", 10000, 1500000, 200000, 5000
            )
        with c3:
            requested_tenure = st.slider("Requested Tenure (months)", 3, 84, 24)

    data = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": float(monthly_salary),
        "employment_type": employment_type,
        "years_of_employment": float(years_of_employment),
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": float(monthly_rent),
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": float(school_fees),
        "college_fees": float(college_fees),
        "travel_expenses": float(travel_expenses),
        "groceries_utilities": float(groceries_utilities),
        "other_monthly_expenses": float(other_monthly_expenses),
        "existing_loans": existing_loans,
        "current_emi_amount": float(current_emi_amount),
        "credit_score": float(credit_score),
        "bank_balance": float(bank_balance),
        "emergency_fund": float(emergency_fund),
        "emi_scenario": emi_scenario,
        "requested_amount": float(requested_amount),
        "requested_tenure": int(requested_tenure),
    }
    return data


def engineer_features_for_pred(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["total_monthly_expenses"] = (
        df[
            [
                "monthly_rent",
                "school_fees",
                "college_fees",
                "travel_expenses",
                "groceries_utilities",
                "other_monthly_expenses",
                "current_emi_amount",
            ]
        ]
        .fillna(0)
        .sum(axis=1)
    )
    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"]
    df["debt_to_income"] = np.where(
        df["monthly_salary"] > 0, df["current_emi_amount"] / df["monthly_salary"], 0
    )
    df["expense_to_income"] = np.where(
        df["monthly_salary"] > 0, df["total_monthly_expenses"] / df["monthly_salary"], 0
    )
    df["affordability_ratio"] = np.where(
        df["monthly_salary"] > 0, df["disposable_income"] / df["monthly_salary"], 0
    )
    df["emi_to_income_req"] = np.where(
        df["monthly_salary"] > 0,
        (df["requested_amount"] / np.maximum(df["requested_tenure"], 1))
        / df["monthly_salary"],
        0,
    )
    df["credit_risk_bin"] = pd.cut(
        df["credit_score"], bins=[0, 550, 650, 750, 900], labels=[3, 2, 1, 0]
    ).astype(float)
    df["employment_stability"] = np.where(
        df["years_of_employment"] >= 5,
        2,
        np.where(df["years_of_employment"] >= 2, 1, 0),
    )
    df["has_existing_loans"] = (df["existing_loans"] == "Yes").astype(int)
    df["savings_months"] = np.where(
        df["monthly_salary"] > 0,
        (df["bank_balance"] + df["emergency_fund"]) / df["monthly_salary"],
        0,
    )
    df["salary_x_credit"] = df["monthly_salary"] * df["credit_score"] / 1000.0
    df["dependents_ratio"] = df["dependents"] / np.maximum(df["family_size"], 1)
    for col in ["monthly_salary", "bank_balance", "emergency_fund", "requested_amount"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


def page_prediction(clf, reg, le, meta):
    st.markdown(
        '<p class="main-header">🔮 Real-time EMI Risk Assessment</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Fill in the applicant profile. Models predict **eligibility** and **maximum safe monthly EMI**."
    )
    data = build_input_form()
    if st.button("🚀 Run Assessment", use_container_width=True):
        with st.spinner("Running models..."):
            feat_df = engineer_features_for_pred(data)
            try:
                pred_class = clf.predict(feat_df)[0]
                proba = clf.predict_proba(feat_df)[0]
                pred_label = le.inverse_transform([pred_class])[0]
                max_emi = max(0.0, float(reg.predict(feat_df)[0]))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.markdown("---")
        st.subheader("📋 Assessment Result")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("**EMI Eligibility**")
            if pred_label == "Eligible":
                st.markdown(
                    f'<p class="eligible">✅ {pred_label}</p>', unsafe_allow_html=True
                )
            elif pred_label == "High_Risk":
                st.markdown(
                    f'<p class="highrisk">⚠️ {pred_label}</p>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p class="noteligible">❌ {pred_label}</p>', unsafe_allow_html=True
                )
        with r2:
            st.metric("Max Safe Monthly EMI", f"₹ {max_emi:,.0f}")
        with r3:
            conf = float(np.max(proba)) * 100
            st.metric("Model Confidence", f"{conf:.1f}%")

        st.markdown("**Class Probabilities**")
        proba_df = pd.DataFrame({"Status": le.classes_, "Probability": proba}).sort_values(
            "Probability", ascending=False
        )
        fig = px.bar(
            proba_df,
            x="Status",
            y="Probability",
            color="Status",
            color_discrete_map={
                "Eligible": "#27ae60",
                "High_Risk": "#f39c12",
                "Not_Eligible": "#e74c3c",
            },
            text=[f"{p*100:.1f}%" for p in proba_df["Probability"]],
        )
        fig.update_layout(showlegend=False, height=320, yaxis_title="Probability", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Recommendation")
        if pred_label == "Eligible":
            st.success(
                f"Applicant appears eligible. Suggested maximum EMI around **₹{max_emi:,.0f}**. Proceed with standard underwriting checks."
            )
        elif pred_label == "High_Risk":
            st.warning(
                f"Marginal case. Consider higher interest / lower amount / shorter tenure. Cap EMI near **₹{max_emi:,.0f}**."
            )
        else:
            st.error(
                f"High risk of default. Loan not recommended at requested terms. Max comfortable EMI ~ **₹{max_emi:,.0f}**."
            )

        st.markdown("**Key Financial Ratios**")
        ratios = {
            "Debt-to-Income": f"{feat_df['debt_to_income'].iloc[0]*100:.1f}%",
            "Expense-to-Income": f"{feat_df['expense_to_income'].iloc[0]*100:.1f}%",
            "Affordability Ratio": f"{feat_df['affordability_ratio'].iloc[0]*100:.1f}%",
            "Savings Buffer (months)": f"{feat_df['savings_months'].iloc[0]:.1f}",
        }
        st.json(ratios)


def page_explorer():
    st.markdown('<p class="main-header">📊 Data Explorer</p>', unsafe_allow_html=True)
    df = load_sample_data(8000)
    if df is None:
        st.warning("Dataset not found.")
        return
    st.write(f"Interactive sample of **{len(df):,}** records (from full 400k set).")
    tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Relationships"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                df,
                names="emi_eligibility",
                color="emi_eligibility",
                color_discrete_map={
                    "Eligible": "#27ae60",
                    "High_Risk": "#f39c12",
                    "Not_Eligible": "#e74c3c",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="emi_scenario", color="emi_eligibility", barmode="group")
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(40), use_container_width=True)
    with tab2:
        col = st.selectbox(
            "Numeric feature",
            [
                "monthly_salary",
                "credit_score",
                "max_monthly_emi",
                "current_emi_amount",
                "bank_balance",
                "requested_amount",
            ],
        )
        fig = px.histogram(
            df,
            x=col,
            color="emi_eligibility",
            marginal="box",
            nbins=40,
            color_discrete_map={
                "Eligible": "#27ae60",
                "High_Risk": "#f39c12",
                "Not_Eligible": "#e74c3c",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        x = st.selectbox(
            "X axis",
            ["monthly_salary", "credit_score", "years_of_employment", "requested_amount"],
        )
        y = st.selectbox("Y axis", ["max_monthly_emi", "current_emi_amount", "bank_balance"])
        fig = px.scatter(
            df.sample(min(2000, len(df))),
            x=x,
            y=y,
            color="emi_eligibility",
            opacity=0.6,
            color_discrete_map={
                "Eligible": "#27ae60",
                "High_Risk": "#f39c12",
                "Not_Eligible": "#e74c3c",
            },
        )
        st.plotly_chart(fig, use_container_width=True)


def page_performance(meta):
    st.markdown('<p class="main-header">📈 Model Performance</p>', unsafe_allow_html=True)
    if meta is None:
        st.warning("Metadata not available.")
        return
    st.subheader("Selected Best Models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Classification (EMI Eligibility)")
        st.write(f"**Best model:** `{meta.get('best_classification_model')}`")
        m = meta.get("class_metrics", {})
        st.metric("Accuracy", f"{m.get('accuracy', 0)*100:.2f}%")
        st.metric("Weighted F1", f"{m.get('f1_weighted', 0)*100:.2f}%")
        st.metric("ROC-AUC (OVR)", f"{m.get('roc_auc', 0):.4f}")
    with c2:
        st.markdown("#### Regression (Max Monthly EMI)")
        st.write(f"**Best model:** `{meta.get('best_regression_model')}`")
        m = meta.get("reg_metrics", {})
        st.metric("RMSE (INR)", f"{m.get('rmse', 0):,.0f}")
        st.metric("MAE (INR)", f"{m.get('mae', 0):,.0f}")
        st.metric("R² Score", f"{m.get('r2', 0):.4f}")
    st.markdown("---")
    st.subheader("Training Summary")
    st.json(
        {
            "Samples used for training": meta.get("n_samples"),
            "Class labels": meta.get("class_labels"),
            "Timestamp": meta.get("timestamp"),
            "Categorical features": len(meta.get("categorical_features", [])),
            "Numerical features": len(meta.get("numerical_features", [])),
        }
    )
    class_rep = REPORTS_DIR / "classification_report.txt"
    if class_rep.exists():
        with st.expander("Full Classification Report"):
            st.code(class_rep.read_text())
    reg_rep = REPORTS_DIR / "regression_metrics.txt"
    if reg_rep.exists():
        with st.expander("Regression Metrics (all models)"):
            st.code(reg_rep.read_text())
    st.subheader("EDA Snapshots")
    cols = st.columns(2)
    img1 = REPORTS_DIR / "eda_eligibility.png"
    img2 = REPORTS_DIR / "eda_correlation.png"
    if img1.exists():
        cols[0].image(str(img1), caption="Eligibility Distribution")
    if img2.exists():
        cols[1].image(str(img2), caption="Feature Correlations")


def page_about():
    st.markdown('<p class="main-header">ℹ️ About EMIPredict AI</p>', unsafe_allow_html=True)
    st.markdown(
        """
    **EMIPredict AI** is an end-to-end financial risk assessment platform that combines:

    - **Classification** models to predict EMI eligibility (Eligible / High_Risk / Not_Eligible)
    - **Regression** models to estimate the maximum safe monthly EMI amount
    - Advanced **feature engineering** (debt-to-income, affordability, savings buffer, etc.)
    - Interactive **Streamlit** web application ready for Streamlit Cloud deployment

    ### Dataset
    - ~400,000 realistic financial profiles
    - 22+ input features covering demographics, income, expenses, credit & loan request
    - 5 EMI scenarios (E-commerce, Appliances, Vehicle, Personal Loan, Education)

    ### Tech Stack
    - Python, Pandas, Scikit-learn, XGBoost
    - Feature engineering & preprocessing pipelines
    - Streamlit + Plotly for UI & interactive charts
    - Joblib for model serialization

    ### Deploy on Streamlit Cloud
    1. Push this repository to GitHub
    2. Go to https://share.streamlit.io
    3. Connect the repo and set main file to `app.py`
    4. Deploy (requirements.txt is included)

    ### Disclaimer
    Educational / demonstration platform only. Real lending decisions must follow 
    regulatory guidelines, full KYC, credit bureau checks and institutional policy.
    """
    )


def main():
    page = sidebar_nav()
    clf, reg, le, meta = load_models()
    if page == "🏠 Home":
        page_home()
    elif page == "🔮 EMI Prediction":
        if clf is None:
            st.error("Models not loaded. Run `python src/train_models.py` first.")
        else:
            page_prediction(clf, reg, le, meta)
    elif page == "📊 Data Explorer":
        page_explorer()
    elif page == "📈 Model Performance":
        page_performance(meta)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
