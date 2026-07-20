"""End-to-end: patient -> reviewable message record.

  lifecycle.assess -> stage + verified claims + triage
  draft.compose (EN & ES) -> message text
  grounding.verify_claim per claim (record + text)  [dual/trajectory grounding]
  if any claim ungrounded: block -> regenerate (feeding the failure back), up to MAX_REGEN;
  still failing -> triage='red', clinician review with the failing claim flagged.

Nothing sends. The output record is what the clinician sees in the review queue.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
from . import lifecycle, draft, grounding, guidelines, config


def _verify_all(pid, base_row, claim_list, text_en) -> list:
    return [grounding.verify_claim(pid, base_row, c, text_en) for c in claim_list]


def _citations(claim_list) -> list:
    refs, seen = [], set()
    for c in claim_list:
        r = c.get("guideline_ref")
        if r and r not in seen:
            seen.add(r)
            ch = guidelines.get_chunk(r)
            if ch:
                refs.append({"id": r, "source": ch["source"]})
    return refs


def build_message(pid: str, base_row: pd.Series, inject_error: bool = False, as_of=None,
                  name: str = None, prev_as_of=None, reenroll: bool = False, flagged=None,
                  enroll_date=None) -> dict:
    """Produce one reviewable message record for a patient.
    inject_error: demo hook, corrupts a numeric fact on the FIRST attempt to show the
    block -> regenerate gate catch it. Regeneration composes cleanly.
    as_of: optional timeline date (walk one patient across lifecycle stages).
    prev_as_of: the patient's previous appointment date (None => enrollment visit).
    reenroll: label a fresh enrollment as a re-enrollment.
    flagged: measures already flagged to the patient (scopes progress/setback and new-OOR).
    enroll_date: date the patient was (re-)enrolled; follow-up staging holds age at this date."""
    a = lifecycle.assess(pid, base_row, as_of=as_of, prev_as_of=prev_as_of, reenroll=reenroll,
                         flagged=flagged, enroll_date=enroll_date)
    a["first_name"] = name
    rec = {**a, "attempts": 0, "message_en": "", "message_es": "",
           "source": "", "grounding": [], "blocked_reasons": [], "citations": _citations(a["claims"]),
           "blocked_message_en": "", "blocked_grounding": []}

    # Clinician-only / no-message stages: nothing to draft.
    if not a["patient_message"] or not a["claims"]:
        rec["message_en"] = "(no patient message, routed to clinician)"
        rec["message_es"] = "(sin mensaje al paciente, enviado al médico)"
        return rec

    claim_list = a["claims"]
    feedback = None
    for attempt in range(1, config.MAX_REGEN + 2):  # first try + up to MAX_REGEN regenerations
        rec["attempts"] = attempt
        text_en, src = draft.compose(a, "en", feedback=feedback)
        text_es, _ = draft.compose(a, "es", feedback=feedback)

        # demo corruption on attempt 1 only
        if inject_error and attempt == 1:
            text_en = _corrupt(text_en, claim_list)

        results = _verify_all(pid, base_row, claim_list, text_en)
        failed = [r for r in results if not r["grounded"]]
        rec.update(message_en=text_en, message_es=text_es, source=src, grounding=results)

        if not failed:
            return rec
        # retain the FIRST blocked draft + its per-claim fact-checks so the UI can show both
        # the caught error and the regenerated clean version
        if not rec["blocked_message_en"]:
            rec["blocked_message_en"] = text_en
            rec["blocked_grounding"] = results
        feedback = "; ".join(f"{r['kind']} claim ({r['failed_layer']}): {r['reason']}" for r in failed)
        rec["blocked_reasons"].append({"attempt": attempt, "reasons": feedback})

    # exhausted regenerations -> route to clinician flagged red
    rec["triage"] = "red"
    rec["triage_reason"] = "grounding failed after regeneration, clinician must verify flagged claims"
    return rec


def _corrupt(text: str, claim_list) -> str:
    """Replace the first claim's first number with a wrong value (demo only)."""
    for c in claim_list:
        if c.get("numbers"):
            n = c["numbers"][0]
            wrong = round(n + 17)
            import re
            return re.sub(rf"\b{int(round(n))}\b", str(wrong), text, count=1)
    return text
