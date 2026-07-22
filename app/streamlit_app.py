"""New England Heart, clinician review queue.

Every message is reviewed here before a (simulated) send; nothing auto-sends. Each row shows the
risk, a diagnostic summary (all model contributions and comorbidities), the bilingual EN/ES draft,
and Approve or Reject. The item badge reflects the action taken.
"""
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neh import config

st.set_page_config(page_title="New England Heart", layout="wide")

BASE_RATE = 0.027
STAGE_LABEL = {"enrollment": "Enrollment", "reenrollment": "Re-enrollment", "progress": "Progress",
               "setback": "Setback", "graduation": "Graduation"}
STAGE_ORDER = {"enrollment": 0, "reenrollment": 0, "progress": 1, "setback": 2, "graduation": 3}
TRIAGE_BADGE = {
    "red": ("GROUNDING FAILED", "#b00020"),
    "normal": ("READY FOR REVIEW", "#1b6b2f"),
}
# Clinical segments for the diagnostic table.
GROUPS = [
    ("Demographics", ["age_at_cutoff", "sex_male"]),
    ("Labs", ["ldl", "hdl", "total_chol", "trig", "hba1c", "egfr"]),
    ("Vitals & Conditions", ["sbp", "dbp", "bmi", "smoking_flag",
                             "diabetes", "ckd", "prior_ascvd", "hypertension"]),
]
# continuous variables get a healthy/unhealthy level: threshold, unhealthy direction
CONT_LEVEL = {"sbp": (130, "high"), "dbp": (80, "high"), "ldl": (100, "high"),
              "hdl": (40, "low"), "total_chol": (200, "high"), "trig": (150, "high"),
              "hba1c": (7.0, "high"), "bmi": (25, "high"), "egfr": (60, "low")}
FLAG_FEATS = {"diabetes", "ckd", "prior_ascvd", "hypertension", "smoking_flag"}
DEMO_FEATS = {"age_at_cutoff", "sex_male"}


@st.cache_data
def load_queue():
    df = pd.read_parquet(config.QUEUE_PARQUET)
    for c in ["drivers", "claims", "grounding", "blocked_grounding", "citations",
              "blocked", "all_contribs", "series", "comorbid"]:
        df[c] = df[f"{c}_json"].apply(json.loads)
    return df


def chip(text, color):
    st.markdown(f"<span style='background:{color};color:white;padding:3px 10px;border-radius:6px;"
                f"font-weight:600;font-size:0.85rem'>{text}</span>", unsafe_allow_html=True)


def title_banner():
    import base64
    from pathlib import Path
    heart_html = ""
    heart_path = Path(__file__).resolve().parents[1] / "figures" / "heart.png"
    if heart_path.exists():
        b64 = base64.b64encode(heart_path.read_bytes()).decode()
        heart_html = (f"<img src='data:image/png;base64,{b64}' "
                      "style='height:2.6rem;margin-left:0.6rem;vertical-align:middle'/>")
    st.markdown(
        "<div style='display:flex;align-items:center;margin:0 0 0.4rem 0'>"
        "<span style='font-size:3rem;font-weight:800;color:#000000;letter-spacing:0.5px'>"
        "New England Heart</span>" + heart_html + "</div>", unsafe_allow_html=True)


def status_of(item_id):
    return st.session_state.get("status", {}).get(item_id)


def set_status(item_id, value, row):
    st.session_state.setdefault("status", {})[item_id] = value
    st.session_state.setdefault("log", [])
    entry = {"action": value, "patient": row["patient"][:8],
             "stage": row["stage"], "sent": value == "SENT"}
    st.session_state["log"].insert(0, entry)
    # persist every approve/reject to the on-disk audit trail (append-only CSV)
    config.ARTIFACTS.mkdir(exist_ok=True)
    pd.DataFrame([{"at": str(pd.Timestamp.now()), **entry}]).to_csv(
        config.AUDIT_LOG, mode="a", header=not config.AUDIT_LOG.exists(), index=False)


def render_badge(row, item_id):
    s = status_of(item_id)
    if s == "SENT":
        chip("SENT", "#1b6b2f")
    elif s == "REJECTED":
        chip("REJECTED", "#b00020")
    else:
        text, color = TRIAGE_BADGE.get(row["triage"], (row["triage"], "#555"))
        chip(text, color)


def _level(feat, value):
    # no level for demographics or yes/no flags
    if value is None or feat in DEMO_FEATS or feat in FLAG_FEATS or feat not in CONT_LEVEL:
        return ""
    thr, direction = CONT_LEVEL[feat]
    if direction == "high":
        return "high" if value >= thr else "normal"
    return "low" if value < thr else "normal"


def _val(feat, value):
    if value is None:
        return ""
    if feat == "sex_male":
        return "male" if value >= 0.5 else "female"
    if feat in FLAG_FEATS:
        return "yes" if value >= 0.5 else "no"
    return f"{round(value, 1)}"


def diagnostic_summary(row):
    st.markdown("**Diagnostic summary**")
    p, thr = row["p_now"], row["threshold"]
    mult = p / BASE_RATE if BASE_RATE else 0
    st.metric("Calibrated 5-year CV risk", f"{p*100:.1f}%",
              delta=f"threshold {thr*100:.1f}%", delta_color="off")
    st.caption(f"Population 5-year event rate is about {BASE_RATE*100:.1f}%, so this is about "
               f"{mult:.1f} times the population rate.")

    contribs = row["all_contribs"]
    by_feat = {c["feature"]: c for c in contribs}
    total = sum(abs(c["shap"]) for c in contribs) or 1.0
    for group_name, feats in GROUPS:
        members = [by_feat[f] for f in feats if f in by_feat]
        if not members:
            continue
        members.sort(key=lambda c: -abs(c["shap"]))
        show_level = group_name != "Demographics"  # no level column for demographics
        rows = []
        for c in members:
            r = {"Variable": c["label"], "Value": _val(c["feature"], c["value"])}
            if show_level:
                r["Level"] = _level(c["feature"], c["value"])
            r["Contribution"] = f"{abs(c['shap'])/total*100:.1f}%"
            rows.append(r)
        st.markdown(f"**{group_name}**")  # bold group header
        st.table(pd.DataFrame(rows))  # st.table renders all rows without scrolling

    com = row["comorbid"]
    flags = [k for k, v in {"diabetes": com.get("diabetes"), "kidney disease": com.get("ckd"),
             "prior cardiovascular disease": com.get("prior_ascvd"),
             "hypertension": com.get("hypertension")}.items() if v]
    st.markdown("**Comorbidities:** " + (", ".join(flags) if flags else "none recorded"))


def citations_block(citations):
    st.markdown("**Guideline citations:**")
    if citations:
        for c in citations:
            st.markdown(f"- {c['id']}: {c['source']}")
    else:
        st.markdown("- none (record-only message)")


def render_detail(row, item_id):
    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Patient {row['patient'][:8]}, {STAGE_LABEL.get(row['stage'], row['stage'])}")
        render_badge(row, item_id)
        diagnostic_summary(row)
        citations_block(row["citations"])

    with right:
        if row.get("blocked_message_en"):
            st.markdown("**Blocked draft (caught by the fact check, not sent):**")
            reasons = "; ".join(b["reasons"] for b in row["blocked"])
            st.markdown(f"<div style='border:2px solid #b00020;border-radius:6px;padding:8px'>"
                        f"{row['blocked_message_en']}</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#b00020;font-weight:600'>Blocked: {reasons}</span>",
                        unsafe_allow_html=True)
            st.markdown("**Regenerated clean draft (below):**")

        st.markdown("**Draft message** (edit inline as needed, then approve)")
        lang = st.radio("Language", ["English", "Espanol"], horizontal=True, key=f"lang_{item_id}")
        text = row["message_en"] if lang == "English" else row["message_es"]
        # widget key includes language so switching languages refreshes the text
        st.text_area("Message", value=text, height=380, key=f"msg_{item_id}_{lang}")

        can_send = row["patient_message"] and row["triage"] != "red"
        c1, c2 = st.columns(2)
        if c1.button("Approve & send", key=f"ap_{item_id}", disabled=not can_send,
                     use_container_width=True):
            set_status(item_id, "SENT", row)
            st.rerun()
        if c2.button("Reject", key=f"rj_{item_id}", use_container_width=True):
            set_status(item_id, "REJECTED", row)
            st.rerun()
        if not can_send:
            st.caption("Sending disabled: routed to a clinician.")


def main():
    title_banner()
    if not config.QUEUE_PARQUET.exists():
        st.error("No queue found. Run: python scripts/precompute_queue.py")
        return
    df = load_queue()

    with st.sidebar:
        st.header("Review queue")
        # single queue view: every reviewable message, grouped patient -> stage (by date)
        main_df = df
        patients = list(dict.fromkeys(main_df["patient"]))

        def plabel(p):
            r = main_df[main_df["patient"] == p]
            name = r.iloc[0].get("first_name") or ""
            return f"{name} ({p[:8]})" if name else p[:8]
        plabels = [plabel(p) for p in patients]
        pi = st.radio("Patient", range(len(plabels)), format_func=lambda i: plabels[i])
        p = patients[pi]
        prows = main_df[main_df["patient"] == p].sort_values("date")  # chronological
        slabels = [f"{STAGE_LABEL.get(r['stage'], r['stage'])} . {r['date']}"
                   for _, r in prows.iterrows()]
        si = st.radio("Stage for this patient", range(len(slabels)),
                      format_func=lambda i: slabels[i], key=f"stage_{p}")
        sel = prows.iloc[si]

        st.divider()
        st.header("Audit log")
        log = st.session_state.get("log", [])
        if log:
            st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)
        else:
            st.caption("No actions yet.")

    render_detail(sel, item_id=str(sel.name))


if __name__ == "__main__":
    main()
