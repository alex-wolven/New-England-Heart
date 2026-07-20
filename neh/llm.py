"""Claude Haiku 4.5 client. Drafting is LLM-only: ANTHROPIC_API_KEY is required
(the offline template mode was removed; see README).

Design: every claim in a message is first computed and verified from the record
(see claims.py). The LLM only does *surface realization* of pre-verified structured
claims and Spanish phrasing -- it never invents clinical facts, and every draft is
re-verified by the grounding gate before it can be queued.
"""
from __future__ import annotations
from typing import Optional
from . import config

_client = None


def available() -> bool:
    return config.LLM_AVAILABLE


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    return _client


def complete(system: str, user: str, model: Optional[str] = None,
             max_tokens: int = 700, temperature: float = 0.3) -> Optional[str]:
    """Return the model's text, or None only when no API key is configured (the caller
    raises a clear missing-key error). Real API failures (network, auth, rate limit) are
    logged and re-raised so they are never mistaken for a missing key."""
    if not available():
        return None
    try:
        resp = _get_client().messages.create(
            model=model or config.DRAFT_MODEL,
            max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    except Exception as e:
        print(f"[llm] API error ({type(e).__name__}: {str(e)[:80]})")
        raise
