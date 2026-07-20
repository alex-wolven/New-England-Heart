"""Precompute the clinician review queue.

Writes artifacts/message_queue.parquet with one row per reviewable message. Selection is fully
automated: we score every cohort patient, walk each candidate's ENTIRE observation timeline to
discover the genuine stages their record supports (enrollment / progress / setback / graduation),
and pick a set of patients with variety in those stages, guaranteeing a few who traverse the
whole journey. Nothing about a patient's stage, value, or date is fabricated; only WHICH real
patients appear is curated. Each row also carries the full clinician-diagnostic payload (all SHAP
contributions, longitudinal BP/LDL/A1c series, comorbidity flags) so the app is a pure viewer.

Requires ANTHROPIC_API_KEY (drafting is LLM-only). Run with `--dry` to print the selected
patients and their stage arcs WITHOUT drafting (no API key or LLM cost).
"""
import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neh import risk, substrate, claims, pipeline, config

COMORBID = ["diabetes", "ckd", "prior_ascvd", "hypertension"]
PATIENT_STAGES = ("enrollment", "reenrollment", "progress", "setback", "graduation")


def _series(pid, when, n=12):
    out = {}
    for m in ("sbp", "dbp", "ldl", "a1c"):
        s = substrate.series(pid, m)
        s = s[s["date"] <= pd.Timestamp(when)].tail(n)
        out[m] = [[str(d.date()), round(float(v), 1)] for d, v in zip(s["date"], s["value"])]
    return out


def _flatten(rec, eval_row, enriched_row, when, extra):
    out = {**extra}
    for k in ("patient", "stage", "triage", "triage_reason", "message_en", "message_es",
              "source", "attempts", "patient_message", "blocked_message_en"):
        out[k] = rec.get(k)
    out["p_now"] = float(rec.get("p_now", 0.0))
    out["threshold"] = float(rec.get("threshold", 0.0))
    out["date"] = str(pd.Timestamp(when).date())
    out["drivers_json"] = json.dumps(rec.get("drivers", []))
    out["claims_json"] = json.dumps(rec.get("claims", []))
    out["grounding_json"] = json.dumps(rec.get("grounding", []))
    out["blocked_grounding_json"] = json.dumps(rec.get("blocked_grounding", []))
    out["citations_json"] = json.dumps(rec.get("citations", []))
    out["blocked_json"] = json.dumps(rec.get("blocked_reasons", []))
    out["all_contribs_json"] = json.dumps(risk.all_contributions(eval_row))
    out["series_json"] = json.dumps(_series(rec["patient"], when))
    out["comorbid_json"] = json.dumps({c: int(enriched_row[c]) for c in COMORBID})
    return out


RAW = Path(r"C:\Users\alexw\OneDrive\Documents\Globus\datathon_dataset")


def _name_map():
    """patient id -> real first name (Synthea appends digits; strip them)."""
    p = pd.read_csv(RAW / "patients.csv", usecols=["Id", "FIRST"], dtype=str)
    return {r.Id: re.sub(r"\d+$", "", str(r.FIRST)).strip() for r in p.itertuples()}


def _timeline(pid):
    """The patient's real checkup dates: actual WELLNESS visits (not every reading). This gives a
    realistic ~annual cadence and is why we don't generate dozens of messages from Synthea's dense
    repeated readings."""
    dates = substrate.wellness_dates(pid)
    return dates[dates >= pd.Timestamp("2013-01-01")]


# --------------------------------------------------------------------------- history discovery
_EPS = 0.002  # risk-change deadband: smaller than this is "flat"


def _history(pid, row):
    """The patient's outreach history at real wellness checkups.

    Enrollment uses a TWO-CHECKUP confirmation (enroll on the second consecutive above-threshold
    visit, when the full panel is usually available) and captures every out-of-range vital there.

    Once enrolled, risk is scored with AGE HELD at the enrollment date, so the trajectory reflects
    the modifiable measures rather than aging. Three follow-up events:
      - new out-of-range measure: a vital outside the healthy range we have not flagged is reported
        immediately (state fact, no buffer), consistent with how enrollment trusts a single reading;
      - progress / setback: a risk move is messaged only when the current move confirms the previous
        one (down then down-or-flat = progress; up then up-or-flat = setback); a reversal seeds the
        next confirmation instead of firing;
      - graduation: age-held risk at/below threshold at two consecutive checkups.
    Returns [(stage, date, prev_date, flagged_frozenset, enroll_date), ...]."""
    thr = risk.enroll_threshold()
    norm = pipeline.lifecycle._norm
    appts, enrolled, ever = [], False, False
    flagged, last_contact, enroll_date = set(), None, None
    prev_risk, pending, below_prev, enroll_pending = None, None, False, False
    for d in _timeline(pid):
        if not enrolled:
            a = pipeline.lifecycle.assess(pid, row, as_of=d, prev_as_of=None, reenroll=ever)
            candidate = a["patient_message"] and a["stage"] in ("enrollment", "reenrollment")
            if candidate and enroll_pending:
                flagged = {norm(v["metric"]) for v in a.get("vitals", [])}  # this visit's panel
                appts.append((a["stage"], d, None, frozenset(flagged), d))
                enrolled, ever, last_contact, enroll_date = True, True, d, d
                pending, enroll_pending = None, False
            else:
                enroll_pending = candidate  # a lone candidate waits for the next checkup to confirm
            r = risk.score_at(row, d)  # real age while scanning for enrollment
            prev_risk, below_prev = r, (r <= thr)
            continue
        # enrolled: age held at enrollment, so the score moves only with the modifiable measures
        r = risk.score_at(row, d, age_ref_date=enroll_date)
        # new out-of-range measure -> immediate setback (state fact), independent of the risk buffer
        if {norm(v["metric"]) for v in claims.out_of_range_vitals(pid, d, {})} - flagged:
            used = frozenset(flagged)  # flagged AS PASSED to assess; main() must replay this exact
            a = pipeline.lifecycle.assess(pid, row, as_of=d, prev_as_of=last_contact,
                                          flagged=flagged, enroll_date=enroll_date)
            if a["patient_message"] and a["stage"] == "setback":
                # store the PRE-update flagged so main()'s re-assessment reproduces this setback
                # (not a graduation); then fold the newly-flagged measures in for later stages
                appts.append(("setback", d, last_contact, used, enroll_date))
                flagged |= set(a.get("newly_flagged", []))
                last_contact = d
            pending, prev_risk, below_prev = None, r, (r <= thr)
            continue
        move = "down" if r < prev_risk - _EPS else ("up" if r > prev_risk + _EPS else "flat")
        # graduation: age-held risk at/below threshold confirmed over two consecutive checkups
        if r <= thr and below_prev:
            a = pipeline.lifecycle.assess(pid, row, as_of=d, prev_as_of=last_contact,
                                          flagged=flagged, enroll_date=enroll_date)
            if a["patient_message"] and a["stage"] == "graduation":
                appts.append(("graduation", d, last_contact, frozenset(flagged), enroll_date))
                enrolled, last_contact, enroll_date = False, d, None
            pending, prev_risk, below_prev = None, r, True
            continue
        # progress / setback: require the current move to confirm the pending direction
        confirmed = ("progress" if pending == "down" and move in ("down", "flat")
                     else "setback" if pending == "up" and move in ("up", "flat") else None)
        if confirmed:
            a = pipeline.lifecycle.assess(pid, row, as_of=d, prev_as_of=last_contact,
                                          flagged=flagged, enroll_date=enroll_date)
            if a["patient_message"] and a["stage"] == confirmed:
                appts.append((confirmed, d, last_contact, frozenset(flagged), enroll_date))
                last_contact = d
            pending = None
        elif move in ("down", "up"):
            pending = move  # reversal / first move seeds the next confirmation
        prev_risk, below_prev = r, (r <= thr)
    return appts


def _select(base, ps, n=10, max_appts=12):
    """Randomly select up to n eligible patients (seeded for reproducibility) whose arc starts at
    enrollment. Random rather than banded: a plain spread across the eligible population, which is
    simpler to state and avoids the banded selection skewing toward the low-risk end."""
    import random
    rng = random.Random(config.SEED)
    pids = list(dict.fromkeys(ps["patient"]))
    rng.shuffle(pids)
    chosen = []
    for pid in pids:
        hist = _history(pid, base.loc[pid])
        if hist and hist[0][0] == "enrollment" and len(hist) <= max_appts:
            chosen.append((pid, hist))
            if len(chosen) >= n:
                break
    return chosen


# --------------------------------------------------------------------------- main
def main(n_patients=10, dry=False):
    df = pd.read_parquet(config.ENRICHED)
    cohort = set(substrate.cohort_patients())
    sub = df[df.patient.isin(cohort)].copy()
    sub["cutoff_date"] = pd.to_datetime(sub["cutoff_date"])
    sub = sub[sub["cutoff_date"] >= pd.Timestamp("2016-01-01")]
    # eligibility: primary prevention only. Do not text patients already on a statin or with
    # established cardiovascular disease (prior ASCVD).
    sub = sub[(sub["on_statin"] == 0) & (sub["prior_ascvd"] == 0)]
    base = risk.patient_base(sub).set_index("patient", drop=False)
    sub_idx = sub.set_index("patient", drop=False)
    sub["p"] = risk.calibrated_proba(base)
    thr = risk.enroll_threshold()
    sub = sub[sub["p"] >= thr]  # only patients the model would enroll

    chosen = _select(base, sub)
    p_by_pid = sub.set_index("patient")["p"]
    print(f"[select] {len(chosen)} eligible patients across the risk spectrum:")
    for pid, hist in chosen:
        print(f"  {pid[:8]} risk={p_by_pid[pid]*100:.1f}% ({len(hist)}): "
              + ", ".join(f"{s}:{d.date()}" for s, d, *_ in hist))
    if dry:
        print("[dry] no drafting performed.")
        return

    names = _name_map()
    rows = []
    for order, (pid, hist) in enumerate(chosen):
        label = names.get(pid)
        for stage, when, prevd, flagged, enroll_date in hist:  # confirmed checkups, chronological
            rec = pipeline.build_message(pid, base.loc[pid], as_of=when, name=label,
                                         prev_as_of=prevd, reenroll=(stage == "reenrollment"),
                                         flagged=set(flagged), enroll_date=enroll_date)
            if not rec.get("patient_message") or rec["stage"] not in PATIENT_STAGES:
                continue
            eval_row = risk.feature_row_at(base.loc[pid], when)
            rows.append(_flatten(rec, eval_row, sub_idx.loc[pid], when,
                                 {"queue": "main", "order": order, "first_name": label}))

    out = pd.DataFrame(rows)
    config.ARTIFACTS.mkdir(exist_ok=True)
    out.to_parquet(config.QUEUE_PARQUET, index=False)
    print(f"[queue] {len(out)} rows across {out['patient'].nunique()} patients "
          f"(avg {len(out)/max(out['patient'].nunique(),1):.1f} appts each)")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
