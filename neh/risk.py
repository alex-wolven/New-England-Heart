"""Risk model: registry-first, race-blind LightGBM + sigmoid (Platt) calibration. Provides SHAP
top-drivers and score-at-date (with optional age-hold) so we can judge the modifiable trajectory
from the longitudinal record.
"""
from __future__ import annotations
import pickle
from functools import lru_cache
from typing import Optional
import numpy as np
import pandas as pd
from . import config, substrate

# substrate metric name -> model feature name
_METRIC2FEAT = {"sbp": "sbp", "dbp": "dbp", "ldl": "ldl", "a1c": "hba1c",
                "total_chol": "total_chol", "hdl": "hdl",
                "bmi": "bmi", "trig": "trig", "egfr": "egfr"}
# human-readable driver labels (for the "why")
DRIVER_LABEL = {
    "age_at_cutoff": "age", "sex_male": "male sex", "diabetes": "diabetes", "ckd": "kidney disease",
    "prior_ascvd": "prior cardiovascular disease", "hypertension": "high blood pressure",
    "ldl": "LDL cholesterol", "sbp": "systolic blood pressure", "dbp": "diastolic blood pressure",
    "bmi": "body-mass index", "smoking_flag": "smoking", "hdl": "HDL cholesterol",
    "total_chol": "total cholesterol", "trig": "triglycerides", "hba1c": "blood sugar (A1c)",
    "egfr": "kidney function (eGFR)",
}
DRIVER_LABEL_ES = {
    "age_at_cutoff": "la edad", "sex_male": "el sexo masculino", "diabetes": "la diabetes",
    "ckd": "la enfermedad renal", "prior_ascvd": "una enfermedad cardiovascular previa",
    "hypertension": "la presión alta", "ldl": "el colesterol LDL",
    "sbp": "la presión sistólica", "dbp": "la presión diastólica",
    "bmi": "el índice de masa corporal", "smoking_flag": "el tabaquismo",
    "hdl": "el colesterol HDL", "total_chol": "el colesterol total",
    "trig": "los triglicéridos", "hba1c": "el azúcar en sangre (A1c)",
    "egfr": "la función renal (eGFR)",
}
NON_MODIFIABLE = {"age_at_cutoff", "sex_male", "prior_ascvd", "ckd"}


def patient_base(df: pd.DataFrame) -> pd.DataFrame:
    """Encoded feature matrix + the columns score-at-date needs (patient, cutoff_date).
    calibrated_proba selects only FEATURES, so the extra columns ride along harmlessly."""
    base = encode_features(df).copy()
    base["patient"] = df["patient"].values
    base["cutoff_date"] = pd.to_datetime(df["cutoff_date"]).values
    return base


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map the enriched cohort columns to the model feature matrix (race-blind)."""
    out = pd.DataFrame(index=df.index)
    out["age_at_cutoff"] = df["age_at_cutoff"].astype(float)
    out["sex_male"] = (df["sex"] == "M").astype(int)
    for c in ["diabetes", "ckd", "prior_ascvd", "hypertension"]:
        out[c] = df[c].astype(int)
    for c in ["ldl", "sbp", "dbp", "bmi", "hdl", "total_chol", "trig", "hba1c", "egfr"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    sm = df["smoking"].astype(str).str.lower()
    out["smoking_flag"] = sm.str.contains("smokes|current").astype(int)
    return out[config.FEATURES]


# --------------------------------------------------------------------------- train
def train_and_save():
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(config.ENRICHED)
    # 5-year label with an observability filter: keep only patients whose 5-year outcome is
    # actually known (a CV event within 5y, OR >=5y of event-free follow-up). Patients censored
    # before 5y are excluded so we never train/score on unknowable outcomes.
    H = config.HORIZON_YEARS
    timing = pd.read_parquet(config.TIMING)[["patient", "event", "time_years"]]
    df = df.merge(timing, on="patient", how="inner")
    df["y_h"] = ((df["event"] == 1) & (df["time_years"] <= H)).astype(int)
    observable = ((df["event"] == 1) & (df["time_years"] <= H)) | (df["time_years"] >= H)
    df = df[observable].copy()
    # registry-first, primary prevention: untreated (no statin) AND no prior CVD, matching the
    # outreach population. The model never sees on_statin or prior_ascvd.
    reg = df[(df["on_statin"] == 0) & (df["prior_ascvd"] == 0)].copy()
    X = encode_features(reg)
    y = reg["y_h"].astype(int).values

    # split by patient (rows are already 1/patient here)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.40, random_state=config.SEED, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=config.SEED, stratify=ytmp)

    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
        class_weight="balanced", random_state=config.SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr)

    # Sigmoid (Platt) calibration on the raw score. Isotonic over-discretized on this
    # rare-event data (~3 probability steps), which zeroed out risk-deltas; sigmoid gives
    # smooth, continuous, monotone probabilities. See LIMITATIONS.
    p_cal_raw = model.predict_proba(Xcal)[:, 1].reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs")
    cal.fit(p_cal_raw, ycal)

    def _apply(raw1d):
        return cal.predict_proba(raw1d.reshape(-1, 1))[:, 1]

    p_te = _apply(model.predict_proba(Xte)[:, 1])
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc = roc_auc_score(yte, p_te)
    brier = brier_score_loss(yte, p_te)

    # enrollment threshold: calibrated risk at the outreach percentile (config) on the cal set
    p_cal = _apply(model.predict_proba(Xcal)[:, 1])
    enroll_threshold = float(np.quantile(p_cal, config.OUTREACH_PERCENTILE))

    artifact = {
        "model": model, "calibrator": cal, "features": config.FEATURES,
        "enroll_threshold": enroll_threshold,
        "meta": {"auc": float(auc), "brier": float(brier), "n_train": int(len(ytr)),
                 "n_registry": int(len(reg)), "event_rate": float(y.mean()),
                 "horizon_years": H, "n_observable": int(len(df))},
    }
    config.ARTIFACTS.mkdir(exist_ok=True)
    with open(config.MODEL_PKL, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[risk] AUC={auc:.3f} Brier={brier:.4f} enroll_threshold={enroll_threshold:.4f} "
          f"registry_n={len(reg):,}")
    return artifact


# --------------------------------------------------------------------------- load / predict
@lru_cache(maxsize=1)
def _artifact():
    with open(config.MODEL_PKL, "rb") as f:
        return pickle.load(f)


def enroll_threshold() -> float:
    return _artifact()["enroll_threshold"]


def calibrated_proba(X: pd.DataFrame) -> np.ndarray:
    a = _artifact()
    Xf = X[a["features"]].astype(float)
    raw = a["model"].predict_proba(Xf)[:, 1].reshape(-1, 1)
    return a["calibrator"].predict_proba(raw)[:, 1]


@lru_cache(maxsize=1)
def _explainer():
    import shap
    return shap.TreeExplainer(_artifact()["model"])


def top_drivers(x_row: pd.Series, k: int = 3):
    """Return the top-k positive risk drivers for one patient as
    [{feature,label,value,shap,modifiable}], most influential first."""
    a = _artifact()
    X = x_row[a["features"]].to_frame().T.astype(float)
    sv = _explainer().shap_values(X)
    vals = sv[1][0] if isinstance(sv, list) else np.asarray(sv)[0]
    order = np.argsort(vals)[::-1]
    drivers = []
    for i in order[:k]:
        f = a["features"][i]
        if vals[i] <= 0:
            continue
        drivers.append({
            "feature": f, "label": DRIVER_LABEL.get(f, f), "label_es": DRIVER_LABEL_ES.get(f, f),
            "value": float(x_row[f]) if pd.notna(x_row[f]) else None,
            "shap": float(vals[i]), "modifiable": f in config.MODIFIABLE,
        })
    return drivers


def all_contributions(x_row: pd.Series):
    """Every feature ranked by absolute SHAP contribution (signed), for the clinician panel:
    [{feature,label,value,shap,modifiable}] most-influential first, positive and negative."""
    a = _artifact()
    X = x_row[a["features"]].to_frame().T.astype(float)
    sv = _explainer().shap_values(X)
    vals = sv[1][0] if isinstance(sv, list) else np.asarray(sv)[0]
    order = np.argsort(np.abs(vals))[::-1]
    out = []
    for i in order:
        f = a["features"][i]
        out.append({
            "feature": f, "label": DRIVER_LABEL.get(f, f),
            "value": float(x_row[f]) if pd.notna(x_row[f]) else None,
            "shap": float(vals[i]), "modifiable": f in config.MODIFIABLE,
        })
    return out


# --------------------------------------------------------------------------- score at date
def feature_row_at(base_row: pd.Series, when: pd.Timestamp, age_ref_date=None) -> pd.Series:
    """Reconstruct the feature vector for a patient as-of `when`: static demographics/
    comorbidities from base_row, longitudinal labs from the substrate as-of that date.

    age_ref_date: if given, age is HELD at the patient's age on that date instead of advanced to
    `when`. This is how the trajectory stages (progress / setback / graduation) isolate the
    MODIFIABLE component of risk: we freeze age at enrollment so the score moves only when the
    coachable measures move. Enrollment/selection passes age_ref_date=None (real age)."""
    x = base_row.copy()
    pid = base_row["patient"]
    cutoff = pd.Timestamp(base_row["cutoff_date"])
    age_date = pd.Timestamp(age_ref_date) if age_ref_date is not None else pd.Timestamp(when)
    years = (cutoff - age_date).days / 365.25
    x["age_at_cutoff"] = float(base_row["age_at_cutoff"]) - max(0.0, years)
    for metric, feat in _METRIC2FEAT.items():
        v = substrate.as_of(pid, metric, when)
        if v is not None:
            x[feat] = v["value"]
    return x


def score_at(base_feats: pd.Series, when: pd.Timestamp, age_ref_date=None) -> float:
    x = feature_row_at(base_feats, when, age_ref_date=age_ref_date)
    return float(calibrated_proba(x.to_frame().T)[0])
