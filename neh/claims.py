"""Structured, record-verifiable claims, the substance of every message.

A claim is a plain dict (JSON/parquet-serializable) with everything needed to (a) realize
it into EN/ES text and (b) INDEPENDENTLY re-verify it against the record + guideline in
grounding.py. Claims never come from the LLM; the LLM only rephrases them. Kinds:

  snapshot     latest value of a metric              -> verify vs latest record value
  change       delta between two appointment dates    -> recompute both readings, compare
  in_range     latest value vs guideline threshold    -> recompute vs threshold (+ guideline)
  actionable   a modifiable driver + guideline rec    -> modifiable feature + guideline chunk exists
  maintenance  general keep-it-up + guideline cite     -> guideline chunk exists

Guideline refs for actionable/in_range/maintenance come from a deterministic metric->passage map
(guidelines.ref_for); other kinds are grounded purely against the record.
"""
from __future__ import annotations
import math
from typing import Optional
import pandas as pd
from . import substrate, guidelines

# physiologically plausible ranges; readings outside are treated as data errors and skipped
PLAUSIBLE = {"sbp": (70, 260), "dbp": (40, 160), "ldl": (20, 400),
             "a1c": (3.5, 20), "total_chol": (80, 500), "hdl": (10, 150),
             "bmi": (12, 80), "trig": (20, 1500), "egfr": (5, 200)}

# Clinically defined healthy ranges (2019 ACC/AHA primary prevention; ADA A1c). Each entry:
# threshold, direction ('high' = unhealthy at/above, 'low' = unhealthy below), healthy text EN/ES,
# and the model feature whose SHAP contribution ranks this vital.
HEALTHY = {
    "bp":         (130, "high", "130/80 or under", "130/80 o menos", "sbp"),
    "ldl":        (100, "high", "100 mg/dL or under", "100 mg/dL o menos", "ldl"),
    "a1c":        (7.0, "high", "7.0% or under", "7.0% o menos", "hba1c"),
    "total_chol": (200, "high", "200 mg/dL or under", "200 mg/dL o menos", "total_chol"),
    "hdl":        (40, "low", "at least 40 mg/dL", "al menos 40 mg/dL", "hdl"),
    # BMI: the healthy TARGET shown to patients is the true healthy line (<25, used for in-range).
    # Weight is only REPORTED at the clinical obesity line (>=30, see _REPORT_THRESHOLD), so we do
    # not message merely-overweight patients.
    "bmi":        (25, "high", "under 25", "menos de 25", "bmi"),
}
# Threshold at which a vital is REPORTED as out of range (out_of_range_vitals). Defaults to the
# healthy threshold; BMI is the exception, only surfaced at the clinical obesity line (>=30).
_REPORT_THRESHOLD = {"bmi": 30.0}
_VITAL_LABEL = {"bp": ("blood pressure", "presión arterial"), "ldl": ("LDL cholesterol", "colesterol LDL"),
                "a1c": ("blood sugar (A1c)", "azúcar en sangre (A1c)"),
                "total_chol": ("total cholesterol", "colesterol total"),
                "hdl": ("HDL cholesterol", "colesterol HDL"),
                "bmi": ("weight (BMI)", "peso (IMC)")}


def out_of_range_vitals(pid: str, when, contrib_by_feat: dict) -> list:
    """Latest values that fall outside the clinical healthy range, ranked by model contribution.
    Each: {metric,label,label_es,value,date,healthy,healthy_es,contribution,numbers}."""
    out = []
    for metric, (thr, direction, he, hs, feat) in HEALTHY.items():
        if metric == "bp":
            s = substrate.as_of(pid, "sbp", when)
            d = substrate.as_of(pid, "dbp", when)
            if not s or not d or not _ok("sbp", s["value"]) or not _ok("dbp", d["value"]):
                continue
            # test DISPLAYED (rounded) values; a value equal to the threshold counts as healthy
            # (unhealthy only strictly above), so out of range means sbp > 130 or dbp > 80
            unhealthy = _rhu(s["value"]) > 130 or _rhu(d["value"]) > 80
            value, date = _fmt_bp(s["value"], d["value"]), max(s["date"], d["date"])
            numbers = [_rhu(s["value"]), _rhu(d["value"])]
        else:
            v = substrate.as_of(pid, metric, when)
            if not v or not _ok(metric, v["value"]):
                continue
            dv = _dnum(metric, v["value"])  # displayed value
            # report threshold: usually the healthy line, but BMI is only surfaced at obesity (30),
            # so we never message merely-overweight patients (the healthy TARGET shown is still <25)
            report_thr = _REPORT_THRESHOLD.get(metric, thr)
            # equal to threshold = healthy: high metrics unhealthy only above, low only below
            unhealthy = (dv > report_thr) if direction == "high" else (dv < report_thr)
            value, date, numbers = _fmt(metric, v["value"]), v["date"], [dv]
        if not unhealthy:
            continue
        lab_en, lab_es = _VITAL_LABEL[metric]
        out.append({"metric": metric, "label": lab_en, "label_es": lab_es, "value": value,
                    "date": str(pd.Timestamp(date).date()), "healthy": he, "healthy_es": hs,
                    "contribution": float(contrib_by_feat.get(feat, 0.0)), "numbers": numbers})
    out.sort(key=lambda x: -x["contribution"])
    return out


def _ok(metric, v) -> bool:
    lo, hi = PLAUSIBLE.get(metric, (-1e9, 1e9))
    return v is not None and lo <= v <= hi


def _rhu(v, nd: int = 0):
    """Round half UP (Python's built-in round is half-to-even, so round(128.5) == 128).
    Used everywhere a value is displayed or stored as a claim number, so display, claim
    numbers, and grounding all agree."""
    f = 10 ** nd
    r = math.floor(float(v) * f + 0.5) / f
    return r if nd else int(r)


def _dnum(metric, v):
    """The number as it is DISPLAYED (so text-faithfulness checks line up exactly)."""
    return _rhu(v, 1) if metric in ("a1c", "bmi") else _rhu(v)


# --------------------------------------------------------------------------- formatting
def _fmt_bp(sbp, dbp) -> str:
    return f"{_rhu(sbp)}/{_rhu(dbp)}"


def _fmt(metric, v) -> str:
    if metric in ("ldl", "total_chol", "hdl"):
        return f"{_rhu(v)} mg/dL"
    if metric == "a1c":
        return f"{_rhu(v, 1):.1f}%"
    if metric == "bmi":
        return f"{_rhu(v, 1):.1f}"
    return f"{_rhu(v)}"


# labels for the generic single-metric snapshot
_SNAP_LABEL = {"ldl": ("LDL cholesterol", "colesterol LDL"),
               "a1c": ("blood sugar (A1c)", "azúcar en sangre (A1c)"),
               "total_chol": ("total cholesterol", "colesterol total"),
               "hdl": ("HDL cholesterol", "colesterol HDL"),
               "bmi": ("weight (BMI)", "peso (IMC)")}


# --------------------------------------------------------------------------- builders
def snapshot_bp(pid: str, when) -> Optional[dict]:
    s = substrate.as_of(pid, "sbp", when)
    d = substrate.as_of(pid, "dbp", when)
    if not s or not d or not _ok("sbp", s["value"]) or not _ok("dbp", d["value"]):
        return None
    bp = _fmt_bp(s["value"], d["value"])
    return {
        "kind": "snapshot", "metric": "bp",
        "text_en": f"your most recent blood pressure was {bp}",
        "text_es": f"su presión arterial más reciente fue {bp}",
        "numbers": [_rhu(s["value"]), _rhu(d["value"])], "direction": None,
        "guideline_ref": None,
        "payload": {"sbp": s["value"], "dbp": d["value"], "when": str(when)},
    }


def snapshot(pid: str, metric: str, when) -> Optional[dict]:
    """Latest single-metric value (ldl / a1c / total_chol / hdl). BP uses snapshot_bp()."""
    if metric not in _SNAP_LABEL:
        return None
    v = substrate.as_of(pid, metric, when)
    if not v or not _ok(metric, v["value"]):
        return None
    label, label_es = _SNAP_LABEL[metric]
    val = v["value"]
    return {
        "kind": "snapshot", "metric": metric,
        "text_en": f"your most recent {label} was {_fmt(metric, val)}",
        "text_es": f"su {label_es} más reciente fue {_fmt(metric, val)}",
        "numbers": [_dnum(metric, val)], "direction": None, "guideline_ref": None,
        "payload": {"value": val, "when": str(pd.Timestamp(when).date()), "metric": metric},
    }


# Threshold to call a change a REAL shift (beyond normal variation), per metric.
#   (value, pct)  pct=True => threshold is a fraction of the previous value (Reference Change
#   Value); pct=False => an absolute change. Every number is cited in DEFENSE.md:
#   LDL/total chol/HDL = biological-variation RCV; A1c = ADA/NICE 0.5 pts; SBP/DBP = minimal
#   detectable change (MDC). BMI, trig, eGFR are NOT messaged (scoring only): trig too labile
#   (~60% RCV), eGFR not a lifestyle target, and BMI is left to the clinician view.
# Threshold to call a change a REAL shift, per metric. (value, pct): pct=True => fraction of the
# previous value (Reference Change Value); pct=False => absolute change. Blood pressure is handled
# as a PAIR ("bp") and fires if EITHER component moves by its MDC (systolic 8, diastolic 12 mmHg).
# Cited in DEFENSE.md. BMI, trig, eGFR are scoring-only (not messaged).
_BP_MDC = {"sbp": 8.0, "dbp": 12.0}
TREND_RULE = {
    "ldl":        (0.20, True),
    "total_chol": (0.17, True),
    "hdl":        (0.18, True),
    "a1c":        (0.5, False),
    "bmi":        (0.05, True),   # ACC/AHA >=5% is a clinically meaningful weight change
    "bp":         (None, False),  # special-cased (pair) in change_since
}
TREND_METRICS = list(TREND_RULE.keys())
# healthy-range threshold per single-value trend metric (direction from _TREND_META); bp is a pair
_TREND_HEALTHY_THR = {"ldl": 100, "a1c": 7.0, "total_chol": 200, "hdl": 40, "bmi": 30}
# unhealthy direction, healthy-range text EN/ES, patient label EN/ES
_TREND_META = {
    "ldl":        ("high", "100 mg/dL or under", "100 mg/dL o menos",
                   "LDL cholesterol", "colesterol LDL"),
    "a1c":        ("high", "7.0% or under", "7.0% o menos", "blood sugar (A1c)",
                   "azúcar en sangre (A1c)"),
    "total_chol": ("high", "200 mg/dL or under", "200 mg/dL o menos",
                   "total cholesterol", "colesterol total"),
    "hdl":        ("low", "at least 40 mg/dL", "al menos 40 mg/dL",
                   "HDL cholesterol", "colesterol HDL"),
    "bmi":        ("high", "under 25", "menos de 25", "weight (BMI)", "peso (IMC)"),
}


def _bp_out(sv, dv):
    """Blood pressure out of range (displayed values); equal to threshold counts as healthy."""
    return _rhu(sv) > 130 or _rhu(dv) > 80


def _change_bp(pid, prev_when, when) -> Optional[dict]:
    """Blood-pressure change reported as the systolic/diastolic PAIR. Fires if either component
    moved by its minimal detectable change (systolic 8, diastolic 12 mmHg) AND the pair is out of
    range at one end. Improvement = total exceedance above 130/80 fell."""
    ps = substrate.as_of(pid, "sbp", pd.Timestamp(prev_when))
    pdb = substrate.as_of(pid, "dbp", pd.Timestamp(prev_when))
    cs = substrate.as_of(pid, "sbp", pd.Timestamp(when))
    cdb = substrate.as_of(pid, "dbp", pd.Timestamp(when))
    if not (ps and pdb and cs and cdb):
        return None
    if not (_ok("sbp", ps["value"]) and _ok("dbp", pdb["value"])
            and _ok("sbp", cs["value"]) and _ok("dbp", cdb["value"])):
        return None
    if abs(cs["value"] - ps["value"]) < _BP_MDC["sbp"] and abs(cdb["value"] - pdb["value"]) < _BP_MDC["dbp"]:
        return None  # neither component moved meaningfully
    if not (_bp_out(ps["value"], pdb["value"]) or _bp_out(cs["value"], cdb["value"])):
        return None  # stayed within the healthy range the whole time

    def _exc(sv, dv):
        return max(_rhu(sv) - 130, 0) + max(_rhu(dv) - 80, 0)
    improved = _exc(cs["value"], cdb["value"]) < _exc(ps["value"], pdb["value"])
    went_down = (_rhu(cs["value"]) + _rhu(cdb["value"])) < (_rhu(ps["value"]) + _rhu(pdb["value"]))
    fv, lv = _fmt_bp(ps["value"], pdb["value"]), _fmt_bp(cs["value"], cdb["value"])
    since = str(pd.Timestamp(ps["date"]).date())
    when_d = str(pd.Timestamp(cs["date"]).date())
    # numeric direction (total mmHg up/down), not good/bad; no date in the bullet
    verb_en = "has dropped from" if went_down else "has increased from"
    verb_es = "ha bajado de" if went_down else "ha subido de"
    return {
        "kind": "change", "metric": "bp",
        "text_en": f"your blood pressure {verb_en} {fv} to {lv} (healthy: 130/80 or under)",
        "text_es": f"su presión arterial {verb_es} {fv} a {lv} (rango saludable: 130/80 o menos)",
        "numbers": [_rhu(ps["value"]), _rhu(pdb["value"]), _rhu(cs["value"]), _rhu(cdb["value"])],
        "direction": "down" if went_down else "up", "guideline_ref": None,
        "payload": {"prev_sbp": ps["value"], "prev_dbp": pdb["value"], "cur_sbp": cs["value"],
                    "cur_dbp": cdb["value"], "metric": "bp", "prev_date": since, "when": when_d,
                    "improved": bool(improved), "dir": "down" if improved else "up"},
    }


def change_since(pid: str, metric: str, prev_when, when) -> Optional[dict]:
    """Change in a metric between the patient's PREVIOUS appointment (prev_when) and now (when),
    comparing the reading as-of each date. Reported only when the change exceeds the metric's
    real-shift threshold (TREND_RULE) AND the measure is out of range at one end. Cites the actual
    baseline reading date. Blood pressure is reported as the systolic/diastolic pair."""
    if metric not in TREND_RULE:
        return None
    if metric == "bp":
        return _change_bp(pid, prev_when, when)
    thr_val, is_pct = TREND_RULE[metric]
    prev = substrate.as_of(pid, metric, pd.Timestamp(prev_when))
    cur = substrate.as_of(pid, metric, pd.Timestamp(when))
    if not prev or not cur or not _ok(metric, prev["value"]) or not _ok(metric, cur["value"]):
        return None
    delta = cur["value"] - prev["value"]
    magnitude = abs(delta) / abs(prev["value"]) if is_pct and prev["value"] else abs(delta)
    if magnitude < thr_val:
        return None
    direction, he, hs, label, label_es = _TREND_META[metric]
    # only message a change when the measure is out of range at one end; a value that stays within
    # the healthy range the whole time is not worth flagging or celebrating
    thr_h = _TREND_HEALTHY_THR[metric]

    def _healthy(v):  # equal to threshold counts as healthy
        return _dnum(metric, v) <= thr_h if direction == "high" else _dnum(metric, v) >= thr_h
    if _healthy(prev["value"]) and _healthy(cur["value"]):
        return None
    improved = (delta < 0) if direction == "high" else (delta > 0)
    fv, lv = _fmt(metric, prev["value"]), _fmt(metric, cur["value"])
    since = str(pd.Timestamp(prev["date"]).date())
    when_d = str(pd.Timestamp(cur["date"]).date())
    # verb reflects the NUMERIC direction (up/down), not whether it is clinically good/bad; no date
    # in the bullet (the date of the last contact is stated once, in the message intro)
    verb_en = "has dropped from" if delta < 0 else "has increased from"
    verb_es = "ha bajado de" if delta < 0 else "ha subido de"
    return {
        "kind": "change", "metric": metric,
        "text_en": f"your {label} {verb_en} {fv} to {lv} (healthy: {he})",
        "text_es": f"su {label_es} {verb_es} {fv} a {lv} (rango saludable: {hs})",
        "numbers": [_dnum(metric, prev["value"]), _dnum(metric, cur["value"])],
        "direction": "down" if delta < 0 else "up",
        "guideline_ref": None,
        "payload": {"prev": prev["value"], "cur": cur["value"], "metric": metric,
                    "prev_date": since, "when": when_d, "improved": bool(improved),
                    "dir": "down" if delta < 0 else "up"},
    }


def in_range(pid: str, metric: str, when) -> Optional[dict]:
    """Latest value is within the healthy range (progress/graduation). Blood pressure ('bp') is the
    systolic/diastolic pair. A value equal to the threshold counts as in range."""
    if metric not in HEALTHY:
        return None
    thr, direction, he, hs, feat = HEALTHY[metric]
    label, label_es = _VITAL_LABEL[metric]
    when_s = str(pd.Timestamp(when).date())
    if metric == "bp":
        s, d = substrate.as_of(pid, "sbp", when), substrate.as_of(pid, "dbp", when)
        if not s or not d or not _ok("sbp", s["value"]) or not _ok("dbp", d["value"]):
            return None
        if not (_rhu(s["value"]) <= 130 and _rhu(d["value"]) <= 80):  # equal = healthy
            return None
        val_str, numbers = _fmt_bp(s["value"], d["value"]), [_rhu(s["value"]), _rhu(d["value"])]
        payload = {"sbp": s["value"], "dbp": d["value"], "metric": "bp", "when": when_s}
    else:
        v = substrate.as_of(pid, metric, when)
        if not v or not _ok(metric, v["value"]):
            return None
        dv = _dnum(metric, v["value"])
        if not (dv <= thr if direction == "high" else dv >= thr):  # equal = healthy
            return None
        val_str, numbers = _fmt(metric, v["value"]), [dv]
        payload = {"value": v["value"], "threshold": thr, "metric": metric, "when": when_s}
    ref = guidelines.ref_for_in_range("sbp" if metric == "bp" else metric)
    return {
        "kind": "in_range", "metric": metric,
        "text_en": f"your {label} is {val_str} (healthy: {he})",
        "text_es": f"su {label_es} es {val_str} (rango saludable: {hs})",
        "numbers": numbers, "direction": None, "guideline_ref": ref, "payload": payload,
    }


def maintenance() -> dict:
    """General keep-up-your-healthy-habits suggestion with a deterministic guideline citation.
    Used on graduation so every message carries a citable guideline reference."""
    return {
        "kind": "maintenance", "metric": "lifestyle",
        "text_en": "keeping up healthy habits helps protect your heart",
        "text_es": "mantener hábitos saludables ayuda a proteger su corazón",
        "numbers": [], "direction": None, "guideline_ref": "lifestyle_general",
        "payload": {},
    }


def actionable(driver: dict) -> Optional[dict]:
    """A modifiable driver + the guideline recommendation it maps to."""
    if not driver.get("modifiable"):
        return None
    feat = driver["feature"]
    labels = {
        "ldl":          ("LDL cholesterol", "colesterol LDL"),
        "total_chol":   ("cholesterol", "colesterol"),
        "hdl":          ("HDL cholesterol", "colesterol HDL"),
        "sbp":          ("blood pressure", "presión arterial"),
        "dbp":          ("blood pressure", "presión arterial"),
        "bmi":          ("weight", "peso"),
        "hba1c":        ("blood sugar", "azúcar en sangre"),
        "smoking_flag": ("smoking", "el tabaquismo"),
    }.get(feat)
    if labels is None:
        return None
    label, label_es = labels
    ref = guidelines.ref_for_actionable(feat)
    # weight is framed around behaviors, never the number, to keep the message supportive
    if feat == "bmi":
        text_en = ("working with your care team on healthy eating and regular activity "
                   "could lower this risk")
        text_es = ("trabajar con su equipo de salud en una alimentación saludable y "
                   "actividad física regular podría reducir este riesgo")
    else:
        text_en = f"working with your care team to improve your {label} could lower this risk"
        text_es = (f"trabajar con su equipo de salud para mejorar su {label_es} "
                   f"podría reducir este riesgo")
    return {
        "kind": "actionable", "metric": feat,
        "text_en": text_en, "text_es": text_es,
        "numbers": [], "direction": None, "guideline_ref": ref,
        "payload": {"feature": feat},
    }
