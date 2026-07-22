"""Compose brief, warm bilingual EN/ES outreach messages.

Every message is assembled deterministically from pre-verified record claims into a fixed layout
(exact greeting line, intro, hyphen bullets, encouragement, closing), so formatting and values are
consistent and correct. Claude then lightly polishes the wording; if it drifts (drops a value,
changes the closing, or alters the bullet count) we fall back to the exact skeleton. Either way the
grounding gate re-checks every number. No age or sex is ever mentioned; no em or en dashes.
"""
from __future__ import annotations
import re
from typing import Optional
from . import llm

_SYS = (
    "You are a warm, upbeat community health outreach writer at about a 6th-grade reading level. "
    "You are given a fully written short SMS. Return it with natural, friendly wording, but you "
    "MUST: keep every number, date, and name exactly; keep each hyphen bullet line exactly as "
    "given; keep the first line and the final line exactly as given; keep the same blank-line "
    "layout; never mention age or sex; never use an em dash or en dash. Do not add or remove lines."
)

_CLOSING = {"en": "Reply anytime to chat with a health professional, or reply STOP to opt out.",
            "es": ("Responda cuando quiera para hablar con un profesional de salud, o responda ALTO "
                   "para no recibir más mensajes.")}

# fixed, approved per-stage lines (EN / ES, formal usted in Spanish)
_OPENER = {  # appended after the greeting on line 1 (empty => greeting alone)
    "enrollment": {"en": "we care about your heart health!", "es": "nos importa su salud del corazón."},
    "reenrollment": {"en": "we're checking in on your heart health again.",
                     "es": "queremos ver de nuevo cómo está su salud del corazón."},
    "progress":   {"en": "You're trending healthier!", "es": "¡Va mejorando!"},
    "graduation": {"en": "You're now out of the elevated risk range, wonderful work!",
                   "es": "¡Ya salió del rango de riesgo elevado, excelente trabajo!"},
    "setback":    {"en": "", "es": ""},
}
# {date} is the patient's last contact date (previous stage), not "last visit": there are
# unmentioned checkups in between, so we anchor to the actual date we last reached out.
_PROGRESS_INTRO = {"en": "Here's what has changed since {date}:",
                   "es": "Esto es lo que ha cambiado desde {date}:"}
_SETBACK_INTRO = {"en": "A few of your health measures have changed since {date}:",
                  "es": "Algunas de sus medidas de salud han cambiado desde {date}:"}
_GRAD_INTRO = {"en": "These measures have moved to the healthy range since {date}:",
               "es": "Estas medidas se han movido al rango saludable desde {date}:"}
_ENCOURAGE = {
    "progress":   {"en": "Your heart risk is still high, so keep up the healthy choices, it's "
                         "making a real difference.",
                   "es": "Su riesgo cardíaco sigue siendo alto, así que siga con las decisiones "
                         "saludables, ¡están marcando una diferencia real!"},
    "graduation": {"en": "Keep up the healthy choices!", "es": "¡Siga con las decisiones saludables!"},
    "enrollment": {"en": "Small steps like eating well and moving more can really help your heart.",
                   "es": "Pequeños pasos como comer bien y moverse más pueden ayudar mucho a su "
                         "corazón."},
    "reenrollment": {"en": "Small steps like eating well and moving more can really help your heart.",
                     "es": "Pequeños pasos como comer bien y moverse más pueden ayudar mucho a su "
                           "corazón."},
    "setback":    {"en": "Small steps like eating well and moving more can really help bring these "
                         "back on track.",
                   "es": "Pequeños pasos como comer bien y moverse más pueden ayudar a mejorar "
                         "estas medidas."},
}


def _greeting(lang, name):
    if name:
        return f"Hola {name}," if lang == "es" else f"Hi {name},"
    return "Hola," if lang == "es" else "Hi there,"


def _enroll_count_intro(n, lang):
    if lang == "es":
        head = {1: "Una de sus medidas de salud recientes está",
                2: "Un par de sus medidas de salud recientes están",
                3: "Algunas de sus medidas de salud recientes están"}.get(
                    n, "Varias de sus medidas de salud recientes están")
        return head + " fuera del rango saludable:"
    head = {1: "One of your recent health measures is",
            2: "A couple of your recent health measures are",
            3: "A few of your recent health measures are"}.get(
                n, "Several of your recent health measures are")
    return head + " outside the healthy range:"


def _shift_intro(n, lang):
    """Setback state intro: the current out-of-range panel, framed as measures that have shifted
    (neutral direction; some may have improved), not the enrollment 'outside the healthy range'."""
    if lang == "es":
        head = {1: "Una de sus medidas recientes ha cambiado",
                2: "Un par de sus medidas recientes han cambiado",
                3: "Algunas de sus medidas recientes han cambiado"}.get(
                    n, "Varias de sus medidas recientes han cambiado")
        return head + ":"
    head = {1: "One of your recent measures has shifted",
            2: "A couple of your recent measures have shifted",
            3: "A few of your recent measures have shifted"}.get(
                n, "Several of your recent measures have shifted")
    return head + ":"


def _bullets(a, lang):
    """The hyphen bullet lines for this stage, exact."""
    stage = a["stage"]
    # enrollment, and a setback that flags a newly out-of-range measure, list vitals as
    # "label: value (healthy: ...)". A change-driven setback/progress lists the change sentences.
    if a.get("vitals") and stage in ("enrollment", "reenrollment", "setback"):
        lk, hk = ("label_es", "healthy_es") if lang == "es" else ("label", "healthy")
        return [f"- {v[lk]}: {v['value']} (healthy: {v[hk]})" for v in a.get("vitals", [])]
    # progress, change-driven setback, and graduation all render "change" claims (the movement)
    return [f"- {c[f'text_{lang}']}" for c in a["claims"] if c["kind"] == "change"]


def _skeleton(a, lang):
    """Assemble the exact message. Returns (text, bullet_lines)."""
    stage = a["stage"]
    greet = _greeting(lang, a.get("first_name"))
    opener = _OPENER[stage][lang]
    line1 = f"{greet} {opener}".strip() if opener else greet
    if stage in ("enrollment", "reenrollment"):
        intro = _enroll_count_intro(len(a.get("vitals", [])), lang)
    elif stage == "progress":
        intro = _PROGRESS_INTRO[lang].format(date=a.get("prev_date") or "")
    elif stage == "setback":
        # vitals present => current out-of-range panel, framed as measures that have SHIFTED (not the
        # enrollment "outside the healthy range"); a change-driven setback reads "changed since {date}"
        intro = (_shift_intro(len(a["vitals"]), lang) if a.get("vitals")
                 else _SETBACK_INTRO[lang].format(date=a.get("prev_date") or ""))
    else:
        intro = _GRAD_INTRO[lang].format(date=a.get("prev_date") or "")
    bullets = _bullets(a, lang)
    encourage = _ENCOURAGE[stage][lang]
    closing = _CLOSING[lang]
    if bullets:
        text = f"{line1}\n\n{intro}\n" + "\n".join(bullets) + f"\n\n{encourage}\n\n{closing}"
    else:  # e.g. a graduation with no previously-flagged measure now in range: no measures block
        text = f"{line1}\n\n{encourage}\n\n{closing}"
    return text, bullets


def _normalize(text: str) -> str:
    """Enforce single blank lines and no blank line between an intro (ends ':') and its bullets."""
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    out = []
    for l in lines:
        if l == "" and (not out or out[-1] == ""):
            continue
        out.append(l)
    while out and out[-1] == "":
        out.pop()
    res = []
    for i, l in enumerate(out):
        if l == "" and res and res[-1].endswith(":") and i + 1 < len(out) and out[i + 1].startswith("-"):
            continue
        res.append(l)
    return "\n".join(res).strip()


def _strip_dashes(text: str) -> str:
    return re.sub(r"\s*–\s*", "-", re.sub(r"\s*—\s*", ", ", text))


def _preserved(txt: str, bullets, must_lines) -> bool:
    """True only if the polished text kept the required fixed lines verbatim and every bullet's
    numbers. Otherwise we use the exact skeleton (guarantees approved wording + values)."""
    for line in must_lines:
        if line and line not in txt:
            return False
    for b in bullets:
        for d in re.findall(r"\d+", b):
            if d not in txt:
                return False
    return True


def compose(a: dict, lang: str, feedback: Optional[str] = None) -> tuple[str, str]:
    """Return (text, 'llm'). Assembles the exact message, has Claude polish it, and falls back to
    the skeleton if the polish drifts. Raises if the API is unavailable."""
    skeleton, bullets = _skeleton(a, lang)
    langname = "Spanish" if lang == "es" else "English"
    prompt = (f"Here is a short outreach SMS in {langname}. Return it with warm, natural, "
              f"encouraging wording while following every rule in the system prompt exactly:\n\n"
              f"{skeleton}")
    if feedback:
        prompt += (f"\n\nA previous draft failed a fact check ({feedback}). Keep every value "
                   f"exactly as written above.")
    txt = llm.complete(_SYS, prompt, temperature=0.3)
    if not txt:
        raise RuntimeError("Drafting requires ANTHROPIC_API_KEY (offline template mode was removed).")
    txt = _normalize(_strip_dashes(txt))
    stage = a["stage"]
    must = [_OPENER[stage][lang], _ENCOURAGE[stage][lang], _CLOSING[lang]]
    if not _preserved(txt, bullets, must):
        txt = _normalize(skeleton)  # guarantee approved wording/values if the model drifted
    return txt, "llm"
