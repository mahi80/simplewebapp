"""
train.py — CC Underwriting Model
Run locally:  python train.py
GitHub Actions runs this automatically on every push.
Model is saved to model/  and tracked in mlruns/
"""
import json, warnings, os, shutil
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from acp import govern

warnings.filterwarnings("ignore")
SEED = 42

# ── MLflow: log everything locally in mlruns/ ─────────────────────────────────
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("cc-underwriting")

with mlflow.start_run(run_name=f"train-{os.getenv('GITHUB_SHA','local')[:7]}"):

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv("cc_underwriting_5k_stratified11.csv")
    print(f"Loaded {df.shape[0]} rows")

    IGNORE = ["applicant_id", "target_approved", "target_credit_limit_assigned"]
    num_cols = [c for c in df.select_dtypes("number").columns if c not in IGNORE]
    cat_cols = [c for c in df.select_dtypes("object").columns if c not in IGNORE]

    # ── Missing values ────────────────────────────────────────────────────────
    df = df.drop(columns=df.columns[df.isnull().mean() > 0.5])
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in num_cols: df[c] = df[c].fillna(df[c].median())
    for c in cat_cols: df[c] = df[c].fillna(df[c].mode()[0])

    # ── Target ────────────────────────────────────────────────────────────────
    df["target"] = (df["target_approved"] == "Yes").astype(int)

    # ── Feature engineering ───────────────────────────────────────────────────
    eps = 1e-6
    df["income_to_limit"]  = df["annual_income"] / (df["requested_credit_limit"] + eps)
    df["expense_ratio"]    = df["total_monthly_expenses"] / (df["monthly_income"] + eps)
    df["bureau_mean"]      = df[["fico_score","equifax_score","experian_score","transunion_score"]].mean(axis=1)
    df["monthly_net"]      = df["monthly_income"] - df["total_monthly_expenses"]
    df["credit_hist_yrs"]  = df["credit_history_length_months"] / 12
    df["age_x_fico"]       = df["age"] * df["fico_score"]
    for c in ["annual_income","net_worth","total_assets","savings_account_balance"]:
        if c in df.columns:
            df[c+"_log"] = np.log1p(df[c].clip(0))

    # Encode categoricals
    cat_ok = [c for c in cat_cols if df[c].nunique() <= 30]
    le = LabelEncoder()
    for c in cat_ok:
        df[c+"_enc"] = le.fit_transform(df[c].astype(str))

    # ── Feature selection ─────────────────────────────────────────────────────
    feature_cols = (num_cols
        + ["income_to_limit","expense_ratio","bureau_mean","monthly_net","credit_hist_yrs","age_x_fico"]
        + [c+"_log" for c in ["annual_income","net_worth","total_assets","savings_account_balance"] if c in df.columns]
        + [c+"_enc" for c in cat_ok])
    feature_cols = [f for f in dict.fromkeys(feature_cols) if f in df.columns]

    # Drop highly correlated
    X_all = df[feature_cols].fillna(0)
    corr = X_all.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    feature_cols = [f for f in feature_cols if f not in drop]
    print(f"Final features: {len(feature_cols)}")

    # ── Split + SMOTE + Scale ─────────────────────────────────────────────────
    X = df[feature_cols].fillna(0)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, y_train = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train ─────────────────────────────────────────────────────────────────
    params = dict(n_estimators=200, max_depth=15, min_samples_split=5, random_state=SEED, n_jobs=-1)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc  = roc_auc_score(y_test, y_prob)
    metrics = {
        "auc":      round(auc, 4),
        "gini":     round(2*auc - 1, 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred, average="weighted"), 4),
    }
    print(f"AUC={metrics['auc']}  Gini={metrics['gini']}  Acc={metrics['accuracy']}  F1={metrics['f1']}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_param("num_features", len(feature_cols))
    mlflow.log_param("git_sha", os.getenv("GITHUB_SHA", "local"))

    # ── Save model + artifacts ────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    if os.path.exists("model/rf"): shutil.rmtree("model/rf")
    mlflow.sklearn.save_model(model, "model/rf")
    # ACP: Wrap model with 35 governance controls before saving
    model = govern(model, "acp_apply_SAL-JOB-06_w7bvn4j1", bu="Sales", budget=10.0)
    with open("model/features.json", "w") as f:
        json.dump(feature_cols, f)
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("model/scaler.json", "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

    try:
        mlflow.log_artifacts("model")
    except Exception as e:
        print(f"Warning: artifact upload skipped ({e})")

    print(f"\n✅ Model saved to model/  |  AUC: {metrics['auc']}")
