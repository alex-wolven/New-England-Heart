"""Central configuration: paths, thresholds, feature lists, guideline constants."""
from __future__ import annotations
import os
from pathlib import Path

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
CHROMA_DIR = ARTIFACTS / "chroma"   # guideline RAG index

ENRICHED = DATA / "neh_cohort_enriched.parquet"
TIMING = DATA / "neh_timing.parquet"   # per-patient event/censoring times

# Outcome horizon. 5 years is the longest horizon this data can validate: ~62% of patients
# have >=5y observed follow-up and 0% have >=10y, so only patients with a 5y event OR >=5y
# event-free follow-up are used for training/eval (the "observable" filter).
HORIZON_YEARS = 5

# The one-time raw scans land in the scratchpad; scripts/build_substrate.py reads
# them and writes the substrate parquets into artifacts/ (git-ignored; regenerate).
SUBSTRATE_LABS = ARTIFACTS / "substrate_labs.parquet"   # patient,date,metric,value
SUBSTRATE_WELLNESS = ARTIFACTS / "substrate_wellness.parquet"  # patient,date (wellness checkups)

MODEL_PKL = ARTIFACTS / "risk_model.pkl"                # dict: model, calibrator, features, meta
QUEUE_PARQUET = ARTIFACTS / "message_queue.parquet"     # precomputed reviewable messages
AUDIT_LOG = ARTIFACTS / "audit_log.csv"                 # visible send/approve/reject log

# --- risk model --------------------------------------------------------------
# Race-blind, primary-prevention: train on the untreated registry with no prior CVD (drop
# on_statin and prior_ascvd; prior_ascvd is constant in this population and carries no model signal).
FEATURES = [
    "age_at_cutoff", "sex_male", "diabetes", "ckd", "hypertension",
    "ldl", "sbp", "dbp", "bmi", "smoking_flag", "hdl", "total_chol", "trig", "hba1c", "egfr",
]
# Modifiable drivers eligible for *actionable* claims (never age/sex/prior events).
MODIFIABLE = {"ldl", "sbp", "dbp", "bmi", "hba1c", "total_chol", "hdl", "smoking_flag"}

# --- outreach threshold ------------------------------------------------------
# Enrollment cut: patients at/above this PERCENTILE of calibrated 5-year risk are eligible.
# 0.80 (top 20%): lifestyle outreach is low-cost/low-harm, so a wider net is guideline-defensible
# and ~doubles reach and event-capture vs the 90th. Baked into the model artifact at train time;
# everything downstream reads risk.enroll_threshold(), so this is the single source of truth.
OUTREACH_PERCENTILE = 0.80

# --- LLM ---------------------------------------------------------------------
DRAFT_MODEL = "claude-haiku-4-5-20251001"
GRADE_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
LLM_AVAILABLE = bool(ANTHROPIC_KEY)
MAX_REGEN = 2   # regeneration attempts before routing to clinician flagged red

SEED = 42
