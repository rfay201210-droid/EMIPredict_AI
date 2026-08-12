# EMIPredict AI – Intelligent Financial Risk Assessment Platform

End-to-end FinTech ML platform for **EMI eligibility classification** and **maximum safe monthly EMI regression**.

## Features

- Dual ML problems: Classification (Eligible / High_Risk / Not_Eligible) + Regression (max_monthly_emi)
- Advanced feature engineering (DTI, expense-to-income, affordability, savings buffer, etc.)
- 5 models each for classification & regression; best models selected automatically
- Multi-page Streamlit application with real-time prediction
- Ready for Streamlit Cloud deployment

## Performance (on held-out test set)

| Task | Best Model | Key Metric |
|------|------------|------------|
| Classification | XGBoost Classifier | Accuracy ≈ 94.3%, Weighted F1 ≈ 92.5%, ROC-AUC ≈ 0.992 |
| Regression | XGBoost Regressor | RMSE ≈ ₹1,256, R² ≈ 0.971 |

## Project Structure

```
EMIPredict_AI/
├── app.py                 # Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   └── EMI_dataset.csv    # ~400k records
├── models/
│   ├── best_classification_model.joblib
│   ├── best_regression_model.joblib
│   ├── label_encoder_eligibility.joblib
│   ├── model_metadata.joblib
│   └── feature_lists.joblib
├── reports/
│   ├── classification_report.txt
│   ├── regression_metrics.txt
│   ├── eda_eligibility.png
│   └── eda_correlation.png
└── src/
    └── train_models.py    # Training + feature engineering pipeline
```

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-train models
python src/train_models.py

# 3. Launch the app
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push the entire `EMIPredict_AI` folder to a GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. New app → select your repo → Main file path: `app.py`
4. Deploy. Streamlit Cloud will install packages from `requirements.txt`.

> Note: The pre-trained models are already included under `models/`. No need to retrain for deployment unless you want to.

## Dataset

- Source: Project-provided EMI dataset (~400,000 rows)
- Features: demographics, employment, income, expenses, credit history, loan request details
- Targets:
  - `emi_eligibility` (3-class)
  - `max_monthly_emi` (continuous)

## Disclaimer

This is an educational demonstration. Production lending systems require full regulatory compliance, credit bureau integration, KYC, and institutional policy overlays.
