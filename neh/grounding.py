"""Trajectory grounding: independently re-verify every claim against the record + guideline,
and check the realized text faithfully renders the claim (catches LLM drift / injected errors).
A claim is grounded only if BOTH pass. Ungrounded -> block; the draft/regenerate loop lives in
pipeline.py (max config.MAX_REGEN attempts, then route to clinician flagged red).
"""
from __future__ import annotations
import re
from typing import Tuple
import pandas as pd
from . import substrate, guidelines, config

_NUM = re.compile(r"\d+\.?\d*")
_DOWN = ["dropped", "down", "decreased", "lower", "bajó", "bajado", "disminuido", "menos"]
_UP = ["risen", "rose", "up", "increased", "higher", "subió", "subido", "aumentado", "más"]


def _nums(text: str):
    return [float(x) for x in _NUM.findall(text)]


# --------------------------------------------------------------------- record verification
def verify_record(pid: str, base_row: pd.Series, claim: dict) -> Tuple[bool, str]:
    """Re-derive the claim from the record. Independent of any generated text."""
    k, p = claim["kind"], claim["payload"]
    try:
        if k == "snapshot" and claim["metric"] == "bp":
            when = p["when"]
            s = substrate.as_of(pid, "sbp", when)
            d = substrate.as_of(pid, "dbp", when)
            if not s or not d:
                return False, "no BP reading at/before date"
            if abs(s["value"] - p["sbp"]) > 1 or abs(d["value"] - p["dbp"]) > 1:
                return False, "BP value does not match record"
            return True, "ok"

        if k == "snapshot":  # generic single-metric snapshot (ldl / a1c / total_chol / hdl)
            m = claim["metric"]
            v = substrate.as_of(pid, m, when=pd.Timestamp(p["when"]))
            tol = 0.1 if m == "a1c" else 1.0
            if not v or abs(v["value"] - p["value"]) > tol:
                return False, f"{m} value does not match record"
            return True, "ok"

        if k == "change" and claim["metric"] == "bp":  # blood-pressure pair change
            for lab, key in (("sbp", "prev_sbp"), ("dbp", "prev_dbp")):
                v = substrate.as_of(pid, lab, pd.Timestamp(p["prev_date"]))
                if not v or abs(v["value"] - p[key]) > 1.5:
                    return False, "BP baseline does not match record"
            for lab, key in (("sbp", "cur_sbp"), ("dbp", "cur_dbp")):
                v = substrate.as_of(pid, lab, pd.Timestamp(p["when"]))
                if not v or abs(v["value"] - p[key]) > 1.5:
                    return False, "BP current value does not match record"
            return True, "ok"

        if k == "change":  # delta between two appointment dates (single metric)
            m = claim["metric"]
            prev = substrate.as_of(pid, m, pd.Timestamp(p["prev_date"]))
            cur = substrate.as_of(pid, m, pd.Timestamp(p["when"]))
            if not prev or not cur:
                return False, "missing reading for change"
            tol = 0.1 if m == "a1c" else 1.5
            if abs(prev["value"] - p["prev"]) > tol or abs(cur["value"] - p["cur"]) > tol:
                return False, "change endpoints do not match record"
            return True, "ok"

        if k == "in_range" and claim["metric"] == "bp":  # blood-pressure pair now in range
            s = substrate.as_of(pid, "sbp", pd.Timestamp(p["when"]))
            d = substrate.as_of(pid, "dbp", pd.Timestamp(p["when"]))
            if not s or not d or not (round(s["value"]) <= 130 and round(d["value"]) <= 80):
                return False, "BP not actually within guideline range"
            if guidelines.get_chunk(claim["guideline_ref"]) is None:
                return False, "guideline reference missing"
            return True, "ok"

        if k == "in_range":  # single metric now in range (equal to threshold counts as in range)
            m = claim["metric"]
            v = substrate.as_of(pid, m, when=pd.Timestamp(p["when"]))
            hi, direction = p["threshold"], ("low" if m == "hdl" else "high")
            ok = v and (v["value"] <= hi + 0.5 if direction == "high" else v["value"] >= hi - 0.5)
            if not ok:
                return False, "value not actually within guideline range"
            if guidelines.get_chunk(claim["guideline_ref"]) is None:
                return False, "guideline reference missing"
            return True, "ok"

        if k == "actionable":
            if guidelines.get_chunk(claim["guideline_ref"]) is None:
                return False, "guideline reference missing"
            if claim["payload"]["feature"] not in config.MODIFIABLE:
                return False, "actionable claim on a non-modifiable feature"
            return True, "ok"

        if k == "maintenance":  # general habit-maintenance suggestion, guideline-cited
            if guidelines.get_chunk(claim["guideline_ref"]) is None:
                return False, "guideline reference missing"
            return True, "ok"
    except Exception as e:
        return False, f"verification error: {type(e).__name__}"
    return False, "unknown claim kind"


# --------------------------------------------------------------------- text faithfulness
def verify_text(claim: dict, text: str) -> Tuple[bool, str]:
    """The realized sentence must contain the claim's numbers and not invert its direction."""
    nums_in_text = _nums(text)
    for expected in claim.get("numbers", []):
        # tolerance 0.6 absorbs integer display rounding (e.g. 139.3 shown as "139")
        if not any(abs(expected - n) <= 0.6 for n in nums_in_text):
            return False, f"text is missing/altered the value {expected}"
    d = claim.get("direction")
    if d:
        low = text.lower()
        opp = _UP if d == "down" else _DOWN
        same = _DOWN if d == "down" else _UP
        if any(w in low for w in opp) and not any(w in low for w in same):
            return False, "text states the opposite direction from the record"
    return True, "ok"


def verify_claim(pid: str, base_row: pd.Series, claim: dict, text: str) -> dict:
    """Dual gate: record AND text. Returns a result dict for the review UI."""
    ok_r, why_r = verify_record(pid, base_row, claim)
    ok_t, why_t = verify_text(claim, text)
    ok = ok_r and ok_t
    reason = "grounded" if ok else (why_r if not ok_r else why_t)
    layer = None if ok else ("record" if not ok_r else "text")
    return {"kind": claim["kind"], "metric": claim["metric"], "grounded": ok,
            "reason": reason, "failed_layer": layer,
            "guideline_ref": claim.get("guideline_ref")}
