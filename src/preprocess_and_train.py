"""
EMIPredict AI - Data Preprocessing, Feature Engineering, Model Training & MLflow Tracking
Handles Classification (EMI Eligibility) and Regression (Max Monthly EMI)
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# MLflow is optional due to environment constraints; we still track experiments manually
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except Exception as e:
    print(f"[Warning] MLflow not fully available ({e}). Continuing without MLflow logging.")
    MLFLOW_AVAILABLE = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
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

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "EMI_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MLRUNS_DIR = BASE_DIR / "mlruns"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

# Set MLflow tracking (if available)
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment("EMIPredict_AI_Experiments")
    except Exception as e:
        print(f"[Warning] Could not set MLflow experiment: {e}")
        MLFLOW_AVAILABLE = False


def load_and_clean_data(sample_size=None, random_state=42):
    """Load and comprehensively clean the dataset."""
    print("=" * 60)
    print("STEP 1: Loading and Cleaning Data")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw shape: {df.shape}")
    
    # Convert problematic columns
    for col in ['age', 'monthly_salary', 'bank_balance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Basic cleaning
    print(f"Missing values before: {df.isnull().sum().sum()}")
    
    # Drop rows where critical targets are missing
    df = df.dropna(subset=['emi_eligibility', 'max_monthly_emi'])
    
    # Fill numerical missing with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")
    
    # Cap extreme outliers for key financial columns (IQR method light)
    for col in ['monthly_salary', 'bank_balance', 'current_emi_amount', 'max_monthly_emi']:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Missing after: {df.isnull().sum().sum()}")
    print(f"Eligibility distribution:\n{df['emi_eligibility'].value_counts(normalize=True)}")
    
    if sample_size and sample_size < len(df):
        # Stratified sample to keep class balance
        df = df.groupby('emi_eligibility', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))), random_state=random_state)
        ).reset_index(drop=True)
        print(f"Sampled shape for training: {df.shape}")
    
    return df


def feature_engineering(df):
    """Create derived financial ratios and risk features."""
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    
    df = df.copy()
    
    # Financial ratios
    df['total_monthly_expenses'] = (
        df['monthly_rent'].fillna(0) +
        df['school_fees'].fillna(0) +
        df['college_fees'].fillna(0) +
        df['travel_expenses'].fillna(0) +
        df['groceries_utilities'].fillna(0) +
        df['other_monthly_expenses'].fillna(0) +
        df['current_emi_amount'].fillna(0)
    )
    
    df['disposable_income'] = df['monthly_salary'] - df['total_monthly_expenses']
    df['debt_to_income'] = np.where(df['monthly_salary'] > 0,
                                    df['current_emi_amount'] / df['monthly_salary'], 0)
    df['expense_to_income'] = np.where(df['monthly_salary'] > 0,
                                       df['total_monthly_expenses'] / df['monthly_salary'], 0)
    df['affordability_ratio'] = np.where(df['monthly_salary'] > 0,
                                         df['disposable_income'] / df['monthly_salary'], 0)
    df['emi_to_income_requested'] = np.where(df['monthly_salary'] > 0,
                                             (df['requested_amount'] / np.maximum(df['requested_tenure'], 1)) / df['monthly_salary'], 0)
    
    # Risk features
    df['credit_risk_score'] = np.where(df['credit_score'] >= 750, 0,
                                       np.where(df['credit_score'] >= 650, 1,
                                                np.where(df['credit_score'] >= 550, 2, 3)))
    df['employment_stability'] = np.where(df['years_of_employment'] >= 5, 2,
                                          np.where(df['years_of_employment'] >= 2, 1, 0))
    df['has_existing_loans'] = (df['existing_loans'] == 'Yes').astype(int)
    df['savings_buffer'] = np.where(df['monthly_salary'] > 0,
                                    (df['bank_balance'] + df['emergency_fund']) / df['monthly_salary'], 0)
    
    # Interaction
    df['salary_x_credit'] = df['monthly_salary'] * df['credit_score'] / 1000
    df['dependents_ratio'] = df['dependents'] / np.maximum(df['family_size'], 1)
    
    # Log transforms for skewed
    for col in ['monthly_salary', 'bank_balance', 'emergency_fund', 'requested_amount']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    print(f"Features after engineering: {df.shape[1]} columns")
    return df


def prepare_features(df):
    """Prepare X, y for both tasks + preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("STEP 3: Preparing Features & Targets")
    print("=" * 60)
    
    # Target encoding for classification
    le_elig = LabelEncoder()
    df['emi_eligibility_encoded'] = le_elig.fit_transform(df['emi_eligibility'])
    
    # Feature columns
    drop_cols = ['emi_eligibility', 'max_monthly_emi', 'emi_eligibility_encoded']
    
    categorical_features = [
        'gender', 'marital_status', 'education', 'employment_type',
        'company_type', 'house_type', 'existing_loans', 'emi_scenario'
    ]
    
    numerical_features = [c for c in df.columns if c not in categorical_features + drop_cols]
    
    # Ensure all exist
    categorical_features = [c for c in categorical_features if c in df.columns]
    numerical_features = [c for c in numerical_features if c in df.columns]
    
    X = df[categorical_features + numerical_features]
    y_class = df['emi_eligibility_encoded']
    y_reg = df['max_monthly_emi']
    
    print(f"X shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y_class).value_counts().to_dict()}")
    print(f"Regression target range: {y_reg.min():.0f} - {y_reg.max():.0f}")
    
    # Preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return X, y_class, y_reg, preprocessor, le_elig, categorical_features, numerical_features


def train_classification_models(X_train, X_test, y_train, y_test, preprocessor, le_elig):
    """Train multiple classification models with MLflow tracking."""
    print("\n" + "=" * 60)
    print("STEP 4: Training Classification Models")
    print("=" * 60)
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42, n_jobs=-1),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost_Classifier": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', eval_metric='mlogloss',
            random_state=42, n_jobs=-1, use_label_encoder=False
        ),
        "Gradient_Boosting_Classifier": GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42),
        "Decision_Tree_Classifier": DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    results = {}
    best_score = -1
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=f"Classification_{name}"):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted') if y_proba is not None else 0.0
            except Exception:
                auc = 0.0
            
            metrics = {
                "accuracy": acc,
                "precision_weighted": prec,
                "recall_weighted": rec,
                "f1_weighted": f1,
                "roc_auc_ovr": auc
            }
            
            mlflow.log_params({f"model": name, **{k: str(v) for k, v in model.get_params().items() if k in ['n_estimators', 'max_depth', 'learning_rate', 'C']}})
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(X_train.head(5), y_pred[:5])
            mlflow.sklearn.log_model(pipeline, "model", signature=signature)
            
            results[name] = {
                "pipeline": pipeline,
                "metrics": metrics,
                "y_pred": y_pred
            }
            
            print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_model_name = name
                best_pipeline = pipeline
    
    print(f"\n>>> Best Classification Model: {best_model_name} (F1={best_score:.4f})")
    
    # Save best
    joblib.dump(best_pipeline, MODELS_DIR / "best_classification_model.joblib")
    joblib.dump(le_elig, MODELS_DIR / "label_encoder_eligibility.joblib")
    
    # Classification report for best
    best_preds = results[best_model_name]["y_pred"]
    report = classification_report(y_test, best_preds, target_names=le_elig.classes_, output_dict=False)
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(report)
    
    return results, best_model_name, best_pipeline


def train_regression_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train multiple regression models with MLflow tracking."""
    print("\n" + "=" * 60)
    print("STEP 5: Training Regression Models")
    print("=" * 60)
    
    models = {
        "Linear_Regression": LinearRegression(n_jobs=-1),
        "Ridge_Regression": Ridge(alpha=1.0, random_state=42),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost_Regressor": XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, objective='reg:squarederror'
        ),
        "Gradient_Boosting_Regressor": GradientBoostingRegressor(n_estimators=80, max_depth=5, random_state=42)
    }
    
    results = {}
    best_score = float('inf')  # lower RMSE better
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=f"Regression_{name}"):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1))) * 100
            
            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape
            }
            
            mlflow.log_params({"model": name})
            mlflow.log_metrics(metrics)
            
            signature = infer_signature(X_train.head(5), y_pred[:5])
            mlflow.sklearn.log_model(pipeline, "model", signature=signature)
            
            results[name] = {
                "pipeline": pipeline,
                "metrics": metrics,
                "y_pred": y_pred
            }
            
            print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")
            
            if rmse < best_score:
                best_score = rmse
                best_model_name = name
                best_pipeline = pipeline
    
    print(f"\n>>> Best Regression Model: {best_model_name} (RMSE={best_score:.2f})")
    
    joblib.dump(best_pipeline, MODELS_DIR / "best_regression_model.joblib")
    
    # Save metrics comparison
    with open(REPORTS_DIR / "regression_metrics.txt", "w") as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        for name, res in results.items():
            f.write(f"{name}:\n")
            for k, v in res["metrics"].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
    
    return results, best_model_name, best_pipeline


def generate_eda_plots(df):
    """Generate key EDA visualizations."""
    print("\n" + "=" * 60)
    print("Generating EDA Plots")
    print("=" * 60)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Eligibility distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['emi_eligibility'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71', '#f39c12'])
    axes[0].set_title('EMI Eligibility Distribution')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=15)
    
    df.groupby('emi_scenario')['emi_eligibility'].value_counts(normalize=True).unstack().plot(
        kind='bar', stacked=True, ax=axes[1], colormap='Set2'
    )
    axes[1].set_title('Eligibility by EMI Scenario')
    axes[1].legend(title='Eligibility', bbox_to_anchor=(1.05, 1))
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_eligibility.png", dpi=120, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap (numerical)
    num_cols = ['monthly_salary', 'credit_score', 'current_emi_amount', 'bank_balance',
                'emergency_fund', 'requested_amount', 'max_monthly_emi', 'debt_to_income',
                'expense_to_income', 'affordability_ratio']
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0, square=True)
    plt.title('Financial Features Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_correlation.png", dpi=120, bbox_inches='tight')
    plt.close()
    
    # 3. Max EMI distribution by eligibility
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in zip(['Eligible', 'High_Risk', 'Not_Eligible'], ['#2ecc71', '#f39c12', '#e74c3c']):
        subset = df[df['emi_eligibility'] == label]['max_monthly_emi']
        if len(subset) > 0:
            ax.hist(subset, bins=40, alpha=0.6, label=label, color=color)
    ax.set_title('Max Monthly EMI Distribution by Eligibility')
    ax.set_xlabel('Max Monthly EMI (INR)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "eda_max_emi_dist.png", dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"EDA plots saved to {REPORTS_DIR}")


def main():
    print("\n" + "#" * 70)
    print("#  EMIPredict AI - Full Training Pipeline")
    print("#  Classification + Regression + MLflow + Feature Engineering")
    print("#" * 70 + "\n")
    
    # Use sample for reasonable training time (can increase to None for full 400k)
    SAMPLE_SIZE = 80000  # ~80k stratified sample - good balance of speed & quality
    
    df = load_and_clean_data(sample_size=SAMPLE_SIZE)
    df = feature_engineering(df)
    generate_eda_plots(df)
    
    X, y_class, y_reg, preprocessor, le_elig, cat_feats, num_feats = prepare_features(df)
    
    # Split - stratified for classification
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Classification
    class_results, best_class_name, best_class_pipe = train_classification_models(
        X_train, X_test, y_class_train, y_class_test, preprocessor, le_elig
    )
    
    # Regression
    reg_results, best_reg_name, best_reg_pipe = train_regression_models(
        X_train, X_test, y_reg_train, y_reg_test, preprocessor
    )
    
    # Save feature lists and metadata
    metadata = {
        "best_classification_model": best_class_name,
        "best_regression_model": best_reg_name,
        "categorical_features": cat_feats,
        "numerical_features": num_feats,
        "class_labels": list(le_elig.classes_),
        "trained_on_samples": len(df),
        "timestamp": datetime.now().isoformat()
    }
    joblib.dump(metadata, MODELS_DIR / "model_metadata.joblib")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Best Classification: {best_class_name}")
    print(f"  Metrics: {class_results[best_class_name]['metrics']}")
    print(f"Best Regression: {best_reg_name}")
    print(f"  Metrics: {reg_results[best_reg_name]['metrics']}")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"MLflow tracking URI: file://{MLRUNS_DIR}")
    print(f"Reports: {REPORTS_DIR}")
    print("\nYou can view MLflow UI with: mlflow ui --backend-store-uri file://" + str(MLRUNS_DIR))
    
    return metadata


if __name__ == "__main__":
    main()
