"""Clinical-guideline RAG over a single source.

Recommendations are grounded by retrieval (Chroma) over passages from one source: the 2019
ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease (Arnett DK et al.,
Circulation 2019;140:e596-e646; PDF in docs/). Retrieval proposes a passage; an allow-set keyed
by (claim_kind, metric) accepts it only if clinically valid, else falls back to a deterministic
pick. Passages below are faithful summaries of that guideline, shown to the clinician.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from . import config

SOURCE = "2019 ACC/AHA Primary Prevention of CVD Guideline (Arnett et al., Circulation 2019;140:e596-e646)"

CHUNKS: List[Dict] = [
    {"id": "acc_bp_target", "source": SOURCE, "topic": "blood pressure target",
     "text": ("For adults with elevated blood pressure or hypertension, a target of generally less "
              "than 130/80 mm Hg is recommended, with nonpharmacologic measures for all such adults "
              "and medication added when indicated. Lowering blood pressure reduces heart attack, "
              "stroke, and heart failure.")},
    {"id": "bp_lifestyle", "source": SOURCE, "topic": "blood pressure and weight lifestyle",
     "text": ("Blood-pressure-lowering steps include a heart-healthy dietary pattern with reduced "
              "sodium, at least 150 minutes per week of moderate-intensity activity, and weight loss "
              "when overweight or obese. A clinically meaningful goal is at least 5 percent weight loss.")},
    {"id": "aha_ldl_target", "source": SOURCE, "topic": "LDL cholesterol lowering",
     "text": ("LDL cholesterol drives atherosclerotic disease and lower is better. A diet emphasizing "
              "vegetables, fruits, legumes, nuts, whole grains, and fish, replacing saturated with "
              "unsaturated fats, lowers LDL; statins lower it further when indicated.")},
    {"id": "statin_primary_prevention", "source": SOURCE, "topic": "who benefits from a statin",
     "text": ("Statin therapy is recommended for adults with LDL 190 mg/dL or higher, adults 40 to 75 "
              "with diabetes, and adults 40 to 75 with a 10-year ASCVD risk of 7.5 percent or higher "
              "after a clinician-patient discussion.")},
    {"id": "statin_faq", "source": SOURCE, "topic": "statins patient-facing",
     "text": ("Statins lower cholesterol and the chance of a heart attack or stroke, and work best "
              "with a heart-healthy diet, activity, and not smoking. Anyone with side effects should "
              "talk with their clinician rather than stopping on their own.")},
    {"id": "ada_a1c_target", "source": SOURCE, "topic": "diabetes and blood sugar",
     "text": ("For type 2 diabetes, a heart-healthy diet and at least 150 minutes per week of activity "
              "are first-line, with metformin when needed. Better long-term blood sugar (a common A1c "
              "goal is under 7.0 percent) lowers complications.")},
    {"id": "lifestyle_general", "source": SOURCE, "topic": "general prevention",
     "text": ("The foundation of prevention is a diet of vegetables, fruits, legumes, whole grains, and "
              "fish; at least 150 minutes per week of moderate activity; a healthy weight; and avoiding "
              "tobacco. Activity also supports healthier cholesterol, including HDL.")},
    {"id": "smoking_cessation", "source": SOURCE, "topic": "smoking",
     "text": ("Tobacco use should be assessed at every visit, and every adult who smokes advised to quit "
              "and offered counseling and support. Cardiovascular risk begins to fall after quitting.")},
]
CHUNK_BY_ID = {c["id"]: c for c in CHUNKS}

# Allow-set: guideline passages clinically valid to attach, by (claim_kind, metric).
ALLOW = {
    ("actionable", "ldl"): ["aha_ldl_target", "statin_primary_prevention", "statin_faq"],
    ("actionable", "total_chol"): ["aha_ldl_target", "statin_primary_prevention", "statin_faq"],
    ("actionable", "hdl"): ["lifestyle_general", "smoking_cessation"],
    ("actionable", "sbp"): ["acc_bp_target", "bp_lifestyle"],
    ("actionable", "dbp"): ["acc_bp_target", "bp_lifestyle"],
    ("actionable", "hba1c"): ["ada_a1c_target"],
    ("actionable", "smoking_flag"): ["smoking_cessation"],
    ("actionable", "bmi"): ["bp_lifestyle", "lifestyle_general"],
    ("in_range", "ldl"): ["aha_ldl_target"],
    ("in_range", "sbp"): ["acc_bp_target"],
    ("in_range", "a1c"): ["ada_a1c_target"],
}
DEFAULT_REF = {
    "ldl": "aha_ldl_target", "total_chol": "aha_ldl_target", "hdl": "lifestyle_general",
    "sbp": "acc_bp_target", "dbp": "acc_bp_target", "hba1c": "ada_a1c_target",
    "a1c": "ada_a1c_target", "smoking_flag": "smoking_cessation", "bmi": "bp_lifestyle",
}
QUERY = {
    "ldl": "lowering LDL cholesterol", "total_chol": "lowering total cholesterol",
    "hdl": "raising HDL through lifestyle and exercise", "sbp": "blood pressure target",
    "dbp": "blood pressure target", "hba1c": "A1c target diabetes", "a1c": "A1c target diabetes",
    "smoking_flag": "quitting smoking heart risk", "bmi": "weight loss healthy lifestyle",
}
_COLLECTION = "neh_guidelines"


def build_index():
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass
    col = client.create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in CHUNKS], documents=[c["text"] for c in CHUNKS],
            metadatas=[{"source": c["source"], "topic": c["topic"]} for c in CHUNKS])
    return col.count()


def _collection():
    import chromadb
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(_COLLECTION)


def retrieve(query: str, k: int = 3) -> List[Dict]:
    try:
        res = _collection().query(query_texts=[query], n_results=k)
        return [{"id": res["ids"][0][i], "text": res["documents"][0][i],
                 "source": res["metadatas"][0][i]["source"]} for i in range(len(res["ids"][0]))]
    except Exception as e:
        print(f"[rag-degraded] Chroma unavailable ({type(e).__name__}), using keyword ranking")
        q = query.lower()
        scored = sorted(CHUNKS, key=lambda c: -sum(w in c["text"].lower() for w in q.split()))
        return [{"id": c["id"], "text": c["text"], "source": c["source"]} for c in scored[:k]]


def resolve_ref(kind: str, metric: str, query: str) -> str:
    """RAG-select a guideline chunk id, guarded by the allow-set; else deterministic fallback."""
    fallback = DEFAULT_REF.get(metric, "lifestyle_general")
    allow = ALLOW.get((kind, metric))
    if not allow:
        return fallback
    hits = retrieve(query, k=1)
    rid = hits[0]["id"] if hits else None
    if rid in allow:
        return rid
    print(f"[rag-fallback] kind={kind} metric={metric} q={query!r} rejected={rid} -> {fallback}")
    return fallback


def ref_for_actionable(feat: str) -> str:
    return resolve_ref("actionable", feat, QUERY.get(feat, feat))


def ref_for_in_range(metric: str) -> str:
    return resolve_ref("in_range", metric, QUERY.get(metric, metric))


def get_chunk(chunk_id: str) -> Optional[Dict]:
    return CHUNK_BY_ID.get(chunk_id)
