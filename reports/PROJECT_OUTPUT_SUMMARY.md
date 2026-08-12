# EMIPredict AI – Project Output Summary

**Generated:** 2026-08-12

## 1. Data Processing

- Source dataset: 404,800 records × 27 columns
- Missing values handled (median for numeric, mode for categorical)
- Duplicates removed, extreme outliers capped (1st–99th percentile)
- Stratified sample of 25,000 records used for training (preserves class distribution)
- Feature engineering produced 16+ derived features:
  - total_monthly_expenses, disposable_income
  - debt_to_income, expense_to_income, affordability_ratio
  - emi_to_income_req, credit_risk_bin, employment_stability
  - has_existing_loans, savings_months, salary_x_credit, dependents_ratio
  - log transforms of key monetary columns

## 2. Classification Results (EMI Eligibility)

| Model                        | Accuracy | Weighted F1 | ROC-AUC |
|-----------------------------|----------|-------------|---------|
| Logistic Regression         | 0.9200   | 0.9001      | 0.9768  |
| Random Forest Classifier    | 0.9334   | 0.9128      | 0.9883  |
| **XGBoost Classifier**      | **0.9426** | **0.9250** | **0.9919** |
| Gradient Boosting Classifier| 0.9374   | 0.9183      | 0.9891  |
| Decision Tree Classifier    | 0.9212   | 0.9242      | 0.9331  |

**Selected model:** XGBoost Classifier  
**Classes:** Eligible, High_Risk, Not_Eligible

## 3. Regression Results (Max Monthly EMI)

| Model                        | RMSE (INR) | MAE (INR) | R²     |
|-----------------------------|------------|-----------|--------|
| Linear Regression           | 3459.1     | 2419.8    | 0.7835 |
| Ridge Regression            | 3459.2     | 2419.8    | 0.7835 |
| Random Forest Regressor     | 1346.8     | 517.4     | 0.9672 |
| **XGBoost Regressor**       | **1255.6** | **541.3** | **0.9715** |
| Gradient Boosting Regressor | 1658.0     | 843.8     | 0.9503 |

**Selected model:** XGBoost Regressor  
Target performance requirement (RMSE < 2000) **achieved**.

## 4. Deliverables Included

- Trained best models (joblib)
- Label encoder & metadata
- Classification report & full regression metrics
- EDA visualizations (eligibility distribution, correlation heatmap)
- Fully functional multi-page Streamlit application (`app.py`)
- `requirements.txt` and `README.md` for Streamlit Cloud deployment

## 5. How to Use

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Cloud: push the folder to GitHub and deploy with main file = `app.py`.
