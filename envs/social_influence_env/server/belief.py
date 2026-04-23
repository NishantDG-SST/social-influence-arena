"""Belief parsing helpers.

Models don't always emit valid JSON. This module provides a robust parser
that extracts a ``BeliefState`` from a raw model completion and falls back
to a sentinel invalid belief if the output can't be recovered. The grader's
INVALID path scores zero on belief-derived signals but still grades the
public response.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from pydantic import ValidationError

from ..models import BeliefState

_BELIEF_JSON_RE = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.DOTALL)


INVALID_BELIEF = BeliefState(
    answer="__INVALID__",
    confidence=0.0,
    reasoning="model output could not be parsed as a BeliefState",
    source="inference",
)


def parse_belief(raw: str) -> Tuple[BeliefState, bool]:
    """Return ``(belief, ok)``. ``ok`` is False when the sentinel was used."""
    if not raw:
        return INVALID_BELIEF, False

    # Direct JSON parse
    try:
        data = json.loads(raw)
        return BeliefState(**data), True
    except (json.JSONDecodeError, ValidationError, TypeError):
        pass

    # Find first JSON-ish blob
    match = _BELIEF_JSON_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return BeliefState(**data), True
        except (json.JSONDecodeError, ValidationError, TypeError):
            pass

    # Best-effort: extract "answer: ..." / confidence lines.
    ans = re.search(r"answer[\"'\s:]*([^,\n}]+)", raw, re.IGNORECASE)
    conf = re.search(r"confidence[\"'\s:]*([01](?:\.\d+)?)", raw, re.IGNORECASE)
    if ans:
        try:
            return (
                BeliefState(
                    answer=ans.group(1).strip().strip('"').strip("'"),
                    confidence=float(conf.group(1)) if conf else 0.5,
                    reasoning="recovered from free-text fallback",
                    source="inference",
                ),
                True,
            )
        except ValidationError:
            pass

    return INVALID_BELIEF, False


def split_completion(raw: str) -> Tuple[Optional[str], str]:
    """Split a model completion into (belief_json, public_response).

    The training prompt asks the model to emit:

        <belief>{...json...}</belief>
        <public>free-text reply</public>

    We're lenient: if tags are missing we try to heuristically split.
    """
    if not raw:
        return None, ""

    m_belief = re.search(r"<belief>(.*?)</belief>", raw, re.DOTALL | re.IGNORECASE)
    m_public = re.search(r"<public>(.*?)</public>", raw, re.DOTALL | re.IGNORECASE)

    if m_belief and m_public:
        return m_belief.group(1).strip(), m_public.group(1).strip()

    # Fall back: find first JSON object, treat the rest as public.
    match = _BELIEF_JSON_RE.search(raw)
    if match:
        belief_json = match.group(0)
        public = (raw[: match.start()] + raw[match.end() :]).strip()
        return belief_json, public

    return None, raw.strip()
