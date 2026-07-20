"""Reach vs events-captured analysis behind the outreach threshold (README "Model" numbers).

Rebuilds the exact seeded train/calibration/test split used by risk.train_and_save, scores the
held-out test set with the EXISTING model artifact (no retraining), and prints, for several
percentile cuts of calibrated risk:
  threshold   the calibrated 5-year risk at that percentile (calibration set)
  reach       share of test patients at/above the threshold (who would be messaged)
  captured    share of all 5-year events in the test set that fall in the messaged group
  group rate  the messaged group's event rate as a multiple of the population rate

The deployed enrollment threshold is config.OUTREACH_PERCENTILE; this script verifies the stored
threshold matches and makes the selection trade-off reproducible from the repo.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neh import config, risk


def main():
    # identical data prep + split to risk.train_and_save (seeded, so it reproduces exactly)
    df = pd.read_parquet(config.ENRICHED)
    H = config.HORIZON_YEARS
    timing = pd.read_parquet(config.TIMING)[["patient", "event", "time_years"]]
    df = df.merge(timing, on="patient", how="inner")
    df["y_h"] = ((df["event"] == 1) & (df["time_years"] <= H)).astype(int)
    observable = ((df["event"] == 1) & (df["time_years"] <= H)) | (df["time_years"] >= H)
    df = df[observable].copy()
    reg = df[(df["on_statin"] == 0) & (df["prior_ascvd"] == 0)].copy()  # primary prevention
    X = risk.encode_features(reg)
    y = reg["y_h"].astype(int).values
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.40,
                                            random_state=config.SEED, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.50,
                                            random_state=config.SEED, stratify=ytmp)

    with open(config.MODEL_PKL, "rb") as f:
        art = pickle.load(f)

    def apply(Xm):
        raw = art["model"].predict_proba(Xm[art["features"]].astype(float))[:, 1].reshape(-1, 1)
        return art["calibrator"].predict_proba(raw)[:, 1]

    p_te, p_cal = apply(Xte), apply(Xcal)
    base_rate = yte.mean()
    print(f"held-out test n={len(yte):,}  5-year event rate={base_rate:.4f}  "
          f"AUC={roc_auc_score(yte, p_te):.4f}")
    dep = config.OUTREACH_PERCENTILE
    q_dep = float(np.quantile(p_cal, dep))
    print(f"stored enroll_threshold={art['enroll_threshold']:.4f}  "
          f"recomputed {dep*100:.0f}th pctile={q_dep:.4f}  "
          f"match={abs(q_dep - art['enroll_threshold']) < 1e-9}")

    print(f"\n{'pctile':>7} {'threshold':>10} {'reach':>7} {'captured':>9} {'group rate':>11}")
    for pct in sorted({0.75, 0.80, 0.85, 0.90, 0.95, dep}):
        t = float(np.quantile(p_cal, pct))
        sel = p_te >= t
        reach, cap = sel.mean(), yte[sel].sum() / yte.sum()
        mult = yte[sel].mean() / base_rate
        mark = "  <-- deployed" if abs(pct - dep) < 1e-9 else ""
        print(f"{pct*100:6.0f}% {t*100:9.2f}% {reach*100:6.1f}% {cap*100:8.1f}% "
              f"{mult:10.1f}x{mark}")


if __name__ == "__main__":
    main()
