"""Message lifecycle: enrollment -> new-out-of-range / progress / setback -> graduation.

We model the lifecycle over each patient's OWN timeline as a series of appointments. The first
appointment (prev_as_of=None) is enrollment: the patient is high risk and we list every measure
outside the healthy range. Each later appointment is assessed relative to the PREVIOUS one
(prev_as_of): a measure newly outside the healthy range is flagged immediately, but only when the
modifiable risk did not fall (a falling risk is progress, never a setback); else if the
modifiable risk (age held at enrollment) has fallen below threshold it is graduation; else we
report the measures that changed by a clinically meaningful amount since the previous visit
(progress if a flagged measure improved, setback if one worsened). Every stage yields
record-verified claims (claims.py). Grounding, not uncertainty, gates sending; nothing sends.
"""
from __future__ import annotations
import pandas as pd
from . import claims, risk

# metric (as used by claims) -> model feature name, for the guideline-cited actionable claim
_METRIC_FEAT = {"bp": "sbp", "sbp": "sbp", "dbp": "dbp", "ldl": "ldl", "a1c": "hba1c",
                "total_chol": "total_chol", "hdl": "hdl", "bmi": "bmi"}
# out-of-range vital metric -> model feature, for the enrollment actionable claim
_VITAL_FEAT = {"bp": "sbp", "ldl": "ldl", "a1c": "hba1c", "total_chol": "total_chol",
               "hdl": "hdl", "bmi": "bmi"}


def _norm(metric):
    """Normalize a metric key for the flagged-measure set: blood pressure is one concept whether
    it was flagged as 'bp' (enrollment) or 'sbp'/'dbp' (a change claim)."""
    return "bp" if metric in ("bp", "sbp", "dbp") else metric


def assess(pid: str, base_row: pd.Series, as_of=None, prev_as_of=None, reenroll=False,
           flagged=None, enroll_date=None) -> dict:
    """Determine stage, build claims + drivers. Returns a reviewable record.
    as_of: date to evaluate the patient at on their own timeline.
    prev_as_of: the patient's previous appointment date. None => this is an enrollment visit;
    a date => a follow-up visit (new-out-of-range / progress / setback / graduation).
    reenroll: label a fresh enrollment as a re-enrollment (the patient graduated earlier and has
    climbed back to high risk).
    flagged: measures already flagged to the patient (enrollment / prior setback / prior new-OOR).
    enroll_date: date the patient was (re-)enrolled. Follow-up staging holds AGE at this date so
    progress/setback/graduation move only with the MODIFIABLE measures, not with aging. Selection
    and enrollment still use real age (enroll_date=None => real age).

    Age policy: age is a driver at ENROLLMENT (it belongs in who we reach out to) but is held
    fixed once enrolled, so we never tell a patient they are doing worse, or deny them progress or
    graduation, merely because they got older."""
    when = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(base_row["cutoff_date"])
    flagged_set = set(flagged) if flagged is not None else set()
    # eval_row for drivers / SHAP / the clinician panel. On a FOLLOW-UP visit (enroll_date set) age
    # is held at enrollment, exactly like the staging score, so age never inflates the displayed
    # contributions stage to stage. Enrollment/selection uses real age (it belongs in who we reach).
    if as_of is not None:
        eval_row = risk.feature_row_at(
            base_row, when, age_ref_date=(enroll_date if prev_as_of is not None else None))
    else:
        eval_row = base_row
    # staging score: age held at enrollment for follow-ups; real age at enrollment/selection.
    if as_of is not None:
        p_now = float(risk.calibrated_proba(
            risk.feature_row_at(base_row, when, age_ref_date=enroll_date).to_frame().T)[0])
    else:
        p_now = float(risk.calibrated_proba(base_row.to_frame().T)[0])
    thr = risk.enroll_threshold()
    drivers = risk.top_drivers(eval_row, k=3)
    contribs = {c["feature"]: abs(c["shap"]) for c in risk.all_contributions(eval_row)}

    def base(stage, triage, reason, claim_list, patient_msg=True):
        return {
            "patient": pid, "stage": stage, "p_now": p_now, "threshold": thr,
            "triage": triage, "triage_reason": reason, "patient_message": patient_msg,
            # date of the patient's last contact (previous stage); stated once in the message intro
            "prev_date": str(pd.Timestamp(prev_as_of).date()) if prev_as_of is not None else None,
            "drivers": drivers, "claims": [c for c in claim_list if c],
        }

    # =========================== follow-up appointment ===========================
    if prev_as_of is not None:
        prev_when = pd.Timestamp(prev_as_of)
        # Modifiable risk at the patient's previous contact (age held at enrollment). Every
        # follow-up decision below is judged against this, so the direction always reflects the
        # coachable trajectory rather than aging.
        p_prev = float(risk.calibrated_proba(
            risk.feature_row_at(base_row, prev_when, age_ref_date=enroll_date).to_frame().T)[0])

        # (1) NEW out-of-range measure: a measure outside the healthy range that we have never
        # flagged. Reported immediately as a state fact (single reading, like enrollment), because
        # being out of range is not visit-to-visit noise. This closes the gap where a measure that
        # is out of range the first time it is ever drawn would otherwise wait for a second reading
        # to form a change. It is a SETBACK only when the modifiable risk did not FALL: if the
        # patient's overall trajectory improved, the visit is progress (handled below), so we never
        # label an improving patient with a setback.
        new_oor = [v for v in claims.out_of_range_vitals(pid, when, contribs)
                   if _norm(v["metric"]) not in flagged_set]
        if new_oor and p_now >= p_prev:
            cl = [claims.snapshot_bp(pid, when) if v["metric"] == "bp"
                  else claims.snapshot(pid, v["metric"], when) for v in new_oor]
            cl.append(claims.actionable({"feature": _VITAL_FEAT[new_oor[0]["metric"]],
                                         "modifiable": True}))
            rec = base("setback", "normal", "a measure is newly outside the healthy range", cl)
            rec["vitals"] = new_oor
            rec["newly_flagged"] = [_norm(v["metric"]) for v in new_oor]
            return rec

        # (2) graduation: modifiable risk (age held at enrollment) has fallen to/below the outreach
        # threshold. Show HOW the flagged measures moved into range (same change logic as
        # progress/setback), plus the guideline-cited maintenance line so a citation always appears.
        if p_now <= thr:
            moves = [c for c in (claims.change_since(pid, m, prev_when, when)
                                 for m in claims.TREND_METRICS) if c]
            # only celebrate flagged measures that IMPROVED and are NOW in the healthy range
            # (a measure that moved but is still out of range does not belong under "moved to the
            # healthy range"); a measure now in range with no meaningful change simply isn't shown
            moves = [c for c in moves
                     if c["payload"]["improved"] and _norm(c["metric"]) in flagged_set
                     and claims.in_range(pid, c["metric"], when) is not None]
            moves.sort(key=lambda c: -contribs.get(_METRIC_FEAT.get(c["metric"], ""), 0.0))
            cl = moves + [claims.maintenance()]
            return base("graduation", "normal", "modifiable risk resolved below threshold", cl)

        # (3) progress / setback. Changes are RCV-confirmed and out-of-range at one end. The veto
        # ("nothing else moved the other way") is scoped to FLAGGED measures, so an unflagged wiggle
        # never silences real news. p_prev (computed above) is the age-held score at the previous
        # contact, so the direction check reflects the modifiable trajectory, not aging.
        changes = [c for c in (claims.change_since(pid, m, prev_when, when)
                               for m in claims.TREND_METRICS) if c]
        changes.sort(key=lambda c: -contribs.get(_METRIC_FEAT.get(c["metric"], ""), 0.0))
        improved = [c for c in changes if c["payload"]["improved"] and _norm(c["metric"]) in flagged_set]
        worsened = [c for c in changes if not c["payload"]["improved"] and _norm(c["metric"]) in flagged_set]

        # progress: a flagged measure improved, none worsened, modifiable risk did not rise
        if improved and not worsened and p_now <= p_prev:
            act = claims.actionable({"feature": _METRIC_FEAT[improved[0]["metric"]],
                                     "modifiable": True})
            return base("progress", "normal", "flagged measures improved and risk did not rise",
                        [c for c in improved + [act] if c])
        # setback: a flagged measure worsened, none improved, modifiable risk did not fall
        if worsened and not improved and p_now >= p_prev:
            act = claims.actionable({"feature": _METRIC_FEAT[worsened[0]["metric"]],
                                     "modifiable": True})
            return base("setback", "normal", "measures worsened and risk did not fall",
                        [c for c in worsened + [act] if c])
        return base("none", "none", "mixed or no clear change since last visit",
                    [], patient_msg=False)

    # =============================== enrollment ==================================
    # Enroll when the patient is above the outreach threshold AND has at least one vital outside
    # the healthy range to coach. Age is allowed to be a driver here (it is part of who we reach).
    if p_now <= thr:
        return base("none", "none", "below outreach threshold", [], patient_msg=False)
    vitals = claims.out_of_range_vitals(pid, when, contribs)
    if not vitals:
        return base("none", "none", "no modifiable vital outside the healthy range",
                    [], patient_msg=False)
    cl = [claims.snapshot_bp(pid, when) if v["metric"] == "bp"
          else claims.snapshot(pid, v["metric"], when) for v in vitals]
    cl.append(claims.actionable({"feature": _VITAL_FEAT[vitals[0]["metric"]], "modifiable": True}))
    stage = "reenrollment" if reenroll else "enrollment"
    rec = base(stage, "normal", "high risk with vitals outside the healthy range", cl)
    rec["vitals"] = vitals
    rec["extra_count"] = 0
    rec["appt_date"] = max(v["date"] for v in vitals)
    return rec
