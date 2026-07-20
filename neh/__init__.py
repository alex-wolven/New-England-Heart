"""New England Heart, uncertainty-gated, trajectory-grounded CV outreach with clinician review.

Package layout:
  config      paths, thresholds, feature lists
  substrate   longitudinal per-patient lab/med series + as-of lookups
  risk        LightGBM train/load, sigmoid (Platt) calibration, ± band, SHAP, score-at-date
  guidelines  curated guideline corpus + Chroma index (retrieval)
  llm         Claude Haiku 4.5 client (ANTHROPIC_API_KEY required; no offline mode)
  claims      compute structured, record-verifiable claims (snapshot/trend/risk-delta/in-range/actionable)
  draft       realize claims into bilingual EN/ES messages
  grounding   re-verify every claim against record + guideline; block -> regenerate (max 2)
  lifecycle   enrollment / progress / graduation staging + review triage
  pipeline    end-to-end: patient -> reviewable message record
"""
__version__ = "0.4.0"
