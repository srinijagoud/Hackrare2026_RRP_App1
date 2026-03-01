
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

DATA_PATH = Path("data/rrp_synthetic.csv")  
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "patient_id"

CATEGORICAL_COLS = [
    "sex",
    "hpv_type",
    "medical_treatment_type",
]

NUMERIC_COLS = [
    "age",
    "immune_compromised",
    "surgeries_last_12m",
    "avg_months_between_surgeries",
    "anatomic_extent",
    "medical_treatment",
    "surgical_treatment",
    # HPO flags (binary numeric)
    "HP_0001609",
    "HP_0010307",
    "HP_0002094",
    "HP_0006536",
    "HP_0012735",
    "HP_0002205",
]

TARGET_MEDICAL = "medical_response"
TARGET_SURGICAL = "surgical_response"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common issues:
    - Convert HP:0002205 -> HP_0002205
    - Ensure all expected HPO columns exist
    - Fill missing values
    """
    df = df.rename(columns={"HP:0002205": "HP_0002205"})


    for col in ["HP_0001609","HP_0010307","HP_0002094","HP_0006536","HP_0012735","HP_0002205"]:
        if col not in df.columns:
            df[col] = 0


    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str)


    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


    if TARGET_MEDICAL in df.columns:
        df[TARGET_MEDICAL] = df[TARGET_MEDICAL].fillna("None").astype(str)
    if TARGET_SURGICAL in df.columns:
        df[TARGET_SURGICAL] = df[TARGET_SURGICAL].fillna("None").astype(str)

    # patient_id can be missing in synthetic; ensure it exists
    if ID_COL not in df.columns:
        df[ID_COL] = [f"ROW{i:05d}" for i in range(len(df))]

    return df

def _make_preprocessor() -> ColumnTransformer:
    """
    OneHot encode categorical columns; pass numeric columns through.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",
    )

def _xgb_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=4,
    )

def _build_pipeline() -> Pipeline:
    pre = _make_preprocessor()
    clf = _xgb_classifier()
    return Pipeline([("pre", pre), ("model", clf)])

def _binary_good_label(series: pd.Series) -> pd.Series:
    """
    Convert 'Good/Partial/Poor/None' to binary:
    Good -> 1, else -> 0
    """
    s = series.astype(str).str.strip().str.lower()
    return (s == "good").astype(int)

def train_and_save_models() -> None:
    print("=== TRAIN START ===")
    print("Working dir:", os.getcwd())
    print("Data path:", DATA_PATH.resolve())
    print("Data exists?:", DATA_PATH.exists())
    print("Model dir:", MODEL_DIR.resolve())

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {DATA_PATH}. "
            f"If you only have rrp_patient_data.csv, either generate expanded CSV "
            f"or change DATA_PATH at top."
        )

    df = pd.read_csv(DATA_PATH)
    df = _normalize_columns(df)

    print("Loaded rows/cols:", df.shape)

    df["high_recurrence"] = (df["surgeries_last_12m"] >= 6).astype(int)

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols].copy()

    y_recur = df["high_recurrence"].copy()

    if TARGET_MEDICAL not in df.columns or TARGET_SURGICAL not in df.columns:
        raise ValueError("Dataset must contain medical_response and surgical_response columns.")

    y_med = _binary_good_label(df[TARGET_MEDICAL])
    y_surg = _binary_good_label(df[TARGET_SURGICAL])

 
    # Train/Test split
    def split_xy(y):
        if y.nunique() >= 2:
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    

    # Train recurrence model
    X_train, X_test, y_train, y_test = split_xy(y_recur)
    recur_pipe = _build_pipeline()
    recur_pipe.fit(X_train, y_train)
    if y_test.nunique() >= 2:
        auc = roc_auc_score(y_test, recur_pipe.predict_proba(X_test)[:, 1])
        print(f"[recurrence] ROC-AUC: {auc:.3f}")
    joblib.dump(recur_pipe, MODEL_DIR / "recurrence_model.pkl")
    print("Saved:", (MODEL_DIR / "recurrence_model.pkl").resolve())


    # Train medical response model
    X_train, X_test, y_train, y_test = split_xy(y_med)
    med_pipe = _build_pipeline()
    med_pipe.fit(X_train, y_train)
    if y_test.nunique() >= 2:
        auc = roc_auc_score(y_test, med_pipe.predict_proba(X_test)[:, 1])
        print(f"[medical_response] ROC-AUC: {auc:.3f}")
    joblib.dump(med_pipe, MODEL_DIR / "medical_response_model.pkl")
    print("Saved:", (MODEL_DIR / "medical_response_model.pkl").resolve())


    # Train surgical response model
    X_train, X_test, y_train, y_test = split_xy(y_surg)
    surg_pipe = _build_pipeline()
    surg_pipe.fit(X_train, y_train)
    if y_test.nunique() >= 2:
        auc = roc_auc_score(y_test, surg_pipe.predict_proba(X_test)[:, 1])
        print(f"[surgical_response] ROC-AUC: {auc:.3f}")
    joblib.dump(surg_pipe, MODEL_DIR / "surgical_response_model.pkl")
    print("Saved:", (MODEL_DIR / "surgical_response_model.pkl").resolve())

    print("=== TRAIN END ===")

if __name__ == "__main__":
    train_and_save_models()