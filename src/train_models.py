"""
EMIPredict AI - Simplified robust training script
Trains Classification + Regression models, saves best ones + reports.
MLflow optional.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "EMI_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_and_clean(sample_size=25000, random_state=42):
    print("=" * 60)
    print("STEP 1: Load & Clean")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw: {df.shape}")

    for col in ['age', 'monthly_salary', 'bank_balance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['emi_eligibility', 'max_monthly_emi'])

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0] if len(df[c].mode()) else 'Unknown')

    df = df.drop_duplicates()

    for col in ['monthly_salary', 'bank_balance', 'current_emi_amount', 'max_monthly_emi']:
        if col in df:
            q01, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q01, q99)

    print(f"Cleaned: {df.shape} | Missing: {df.isnull().sum().sum()}")
    print(df['emi_eligibility'].value_counts(normalize=True).round(3))

    if sample_size and sample_size < len(df):
        fractions = df['emi_eligibility'].value_counts(normalize=True)
        pieces = []
        for label, frac in fractions.items():
            n = max(1, int(sample_size * frac))
            subset = df[df['emi_eligibility'] == label]
            pieces.append(subset.sample(n=min(n, len(subset)), random_state=random_state))
        df = pd.concat(pieces, ignore_index=True)
        print(f"Sampled: {df.shape}")
    return df


def feature_engineering(df):
    print("\nSTEP 2: Feature Engineering")
    df = df.copy()
    df['total_monthly_expenses'] = (
        df[['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses',
            'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']].fillna(0).sum(axis=1)
    )
    df['disposable_income'] = df['monthly_salary'] - df['total_monthly_expenses']
    df['debt_to_income'] = np.where(df['monthly_salary'] > 0, df['current_emi_amount'] / df['monthly_salary'], 0)
    df['expense_to_income'] = np.where(df['monthly_salary'] > 0, df['total_monthly_expenses'] / df['monthly_salary'], 0)
    df['affordability_ratio'] = np.where(df['monthly_salary'] > 0, df['disposable_income'] / df['monthly_salary'], 0)
    df['emi_to_income_req'] = np.where(
        df['monthly_salary'] > 0,
        (df['requested_amount'] / np.maximum(df['requested_tenure'], 1)) / df['monthly_salary'], 0
    )
    df['credit_risk_bin'] = pd.cut(df['credit_score'], bins=[0, 550, 650, 750, 900], labels=[3, 2, 1, 0]).astype(float)
    df['employment_stability'] = np.where(df['years_of_employment'] >= 5, 2, np.where(df['years_of_employment'] >= 2, 1, 0))
    df['has_existing_loans'] = (df['existing_loans'] == 'Yes').astype(int)
    df['savings_months'] = np.where(df['monthly_salary'] > 0, (df['bank_balance'] + df['emergency_fund']) / df['monthly_salary'], 0)
    df['salary_x_credit'] = df['monthly_salary'] * df['credit_score'] / 1000.0
    df['dependents_ratio'] = df['dependents'] / np.maximum(df['family_size'], 1)
    for col in ['monthly_salary', 'bank_balance', 'emergency_fund', 'requested_amount']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    print(f"Engineered shape: {df.shape}")
    return df


def prepare(df):
    print("\nSTEP 3: Prepare X/y")
    le = LabelEncoder()
    df['y_class'] = le.fit_transform(df['emi_eligibility'])

    cat_feats = ['gender', 'marital_status', 'education', 'employment_type',
                 'company_type', 'house_type', 'existing_loans', 'emi_scenario']
    cat_feats = [c for c in cat_feats if c in df.columns]

    drop = ['emi_eligibility', 'max_monthly_emi', 'y_class']
    num_feats = [c for c in df.columns if c not in cat_feats + drop and df[c].dtype != 'object']

    X = df[cat_feats + num_feats]
    y_c = df['y_class']
    y_r = df['max_monthly_emi']

    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)])

    print(f"X: {X.shape} | Classes: {dict(zip(*np.unique(y_c, return_counts=True)))}")
    return X, y_c, y_r, pre, le, cat_feats, num_feats


def train_class(Xtr, Xte, ytr, yte, pre, le):
    print("\n" + "=" * 60)
    print("STEP 4: Classification Models")
    print("=" * 60)
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=800, random_state=42, n_jobs=-1),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost_Classifier": XGBClassifier(n_estimators=40, max_depth=5, learning_rate=0.1,
                                            objective='multi:softprob', eval_metric='mlogloss',
                                            random_state=42, n_jobs=-1),
        "Gradient_Boosting_Classifier": GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42),
        "Decision_Tree_Classifier": DecisionTreeClassifier(max_depth=10, random_state=42),
    }
    results = {}
    best_f1, best_name, best_pipe = -1, None, None

    for name, clf in models.items():
        print(f"  Training {name} ...", end=" ")
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte) if hasattr(pipe, 'predict_proba') else None
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average='weighted', zero_division=0)
        prec = precision_score(yte, pred, average='weighted', zero_division=0)
        rec = recall_score(yte, pred, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(yte, proba, multi_class='ovr', average='weighted') if proba is not None else 0
        except Exception:
            auc = 0.0
        metrics = dict(accuracy=acc, f1_weighted=f1, precision=prec, recall=rec, roc_auc=auc)
        results[name] = dict(pipe=pipe, metrics=metrics, pred=pred)
        print(f"Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")
        if f1 > best_f1:
            best_f1, best_name, best_pipe = f1, name, pipe

    print(f"\n>>> BEST CLASSIFICATION: {best_name} (F1={best_f1:.4f})")
    joblib.dump(best_pipe, MODELS_DIR / "best_classification_model.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder_eligibility.joblib")

    report = classification_report(yte, results[best_name]['pred'], target_names=le.classes_)
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Best Model: {best_name}\n\n{report}\n\nAll metrics:\n")
        for n, r in results.items():
            f.write(f"{n}: {r['metrics']}\n")
    return results, best_name, best_pipe


def train_reg(Xtr, Xte, ytr, yte, pre):
    print("\n" + "=" * 60)
    print("STEP 5: Regression Models")
    print("=" * 60)
    models = {
        "Linear_Regression": LinearRegression(n_jobs=-1),
        "Ridge_Regression": Ridge(alpha=1.0, random_state=42),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=40, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost_Regressor": XGBRegressor(n_estimators=40, max_depth=5, learning_rate=0.1,
                                          random_state=42, n_jobs=-1, objective='reg:squarederror'),
        "Gradient_Boosting_Regressor": GradientBoostingRegressor(n_estimators=30, max_depth=4, random_state=42),
    }
    results = {}
    best_rmse, best_name, best_pipe = float('inf'), None, None

    for name, reg in models.items():
        print(f"  Training {name} ...", end=" ")
        pipe = Pipeline([('pre', pre), ('reg', reg)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        mae = mean_absolute_error(yte, pred)
        r2 = r2_score(yte, pred)
        mape = np.mean(np.abs((yte - pred) / np.maximum(np.abs(yte), 1))) * 100
        metrics = dict(rmse=rmse, mae=mae, r2=r2, mape=mape)
        results[name] = dict(pipe=pipe, metrics=metrics, pred=pred)
        print(f"RMSE={rmse:.1f} MAE={mae:.1f} R2={r2:.4f}")
        if rmse < best_rmse:
            best_rmse, best_name, best_pipe = rmse, name, pipe

    print(f"\n>>> BEST REGRESSION: {best_name} (RMSE={best_rmse:.1f})")
    joblib.dump(best_pipe, MODELS_DIR / "best_regression_model.joblib")

    with open(REPORTS_DIR / "regression_metrics.txt", "w") as f:
        f.write(f"Best Model: {best_name}\n\n")
        for n, r in results.items():
            f.write(f"{n}:\n")
            for k, v in r['metrics'].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
    return results, best_name, best_pipe


def eda_plots(df):
    print("\nGenerating EDA plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    df['emi_eligibility'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#27ae60', '#f39c12'])
    axes[0].set_title('EMI Eligibility Distribution')
    axes[0].tick_params(axis='x', rotation=15)
    (df.groupby('emi_scenario')['emi_eligibility']
       .value_counts(normalize=True).unstack()
       .plot(kind='bar', stacked=True, ax=axes[1], colormap='Set2'))
    axes[1].set_title('Eligibility Share by Scenario')
    axes[1].legend(title='Status', bbox_to_anchor=(1.02, 1), fontsize=8)
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_eligibility.png", dpi=110, bbox_inches='tight')
    plt.close()

    cols = [c for c in ['monthly_salary', 'credit_score', 'current_emi_amount', 'bank_balance',
                        'emergency_fund', 'requested_amount', 'max_monthly_emi',
                        'debt_to_income', 'expense_to_income', 'affordability_ratio'] if c in df]
    if len(cols) > 3:
        plt.figure(figsize=(9, 7))
        sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='RdYlGn', center=0, square=True, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "eda_correlation.png", dpi=110, bbox_inches='tight')
        plt.close()
    print(f"Plots -> {REPORTS_DIR}")


def main():
    print("\n" + "#" * 65)
    print("#  EMIPredict AI  |  Training Pipeline")
    print("#" * 65)

    df = load_and_clean(sample_size=25000)
    df = feature_engineering(df)
    eda_plots(df)

    X, yc, yr, pre, le, catf, numf = prepare(df)
    Xtr, Xte, yctr, ycte, yrtr, yrte = train_test_split(
        X, yc, yr, test_size=0.2, random_state=42, stratify=yc
    )
    print(f"Train: {len(Xtr)} | Test: {len(Xte)}")

    cres, cbest, cpipe = train_class(Xtr, Xte, yctr, ycte, pre, le)
    rres, rbest, rpipe = train_reg(Xtr, Xte, yrtr, yrte, pre)

    meta = {
        "best_classification_model": cbest,
        "best_regression_model": rbest,
        "class_metrics": cres[cbest]['metrics'],
        "reg_metrics": rres[rbest]['metrics'],
        "categorical_features": catf,
        "numerical_features": numf,
        "class_labels": list(le.classes_),
        "n_samples": len(df),
        "timestamp": datetime.now().isoformat()
    }
    joblib.dump(meta, MODELS_DIR / "model_metadata.joblib")

    # Also save a sample of raw feature columns for the app
    joblib.dump({"cat": catf, "num": numf}, MODELS_DIR / "feature_lists.joblib")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Best Class : {cbest} -> {cres[cbest]['metrics']}")
    print(f"Best Reg   : {rbest} -> {rres[rbest]['metrics']}")
    print(f"Artifacts  : {MODELS_DIR}")
    print("=" * 60)
    return meta


if __name__ == "__main__":
    main()
