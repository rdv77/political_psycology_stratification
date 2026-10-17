"""Pure lexical mention guard for relevance results, not identity verification.

A name/alias match is necessary for automatic relevance under this policy, but
does not prove that a homonym or shared name component identifies the actor.
Apply after result/schema validation. No source or history is modified.
"""
from __future__ import annotations

import copy
import re
import unicodedata


IDENTITY_POLICY_VERSION = "lexical-mention-review-v1"


def _lexical_text(value: str) -> str:
    # Preserve accents and punctuation; only canonical Unicode, case and
    # whitespace differ. No inferred aliases, surnames or transliterations.
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def ground_relevance(result: dict, context: dict) -> dict:
    """Copy a result, downgrading relevant to review if any actor lacks a mention.

    Context uses document.text and config.objects[{id, name, aliases?}], as in
    corpus models. Only supplied approved labels are checked with Unicode word
    boundaries. Author metadata, roles and pronouns do not supply a name match.
    The original actor_ids and reason remain available for diagnostic review.
    """
    grounded = copy.deepcopy(result)
    if grounded.get("status") != "relevant":
        return grounded
    actors = {actor["id"]: actor for actor in context.get("config", {}).get("objects", [])}
    text = _lexical_text(context.get("document", {}).get("text", ""))
    missing = []
    for actor_id in grounded.get("actor_ids", []):
        actor = actors.get(actor_id, {})
        labels = [actor.get("name", ""), *actor.get("aliases", [])]
        if not any(re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text)
                   for label in (_lexical_text(label) for label in labels) if label):
            missing.append(actor_id)
    if missing:
        grounded["status"] = "review"
        explanation = ("Identity guard: no approved name/alias mention in document.text for actor IDs: "
                       + ", ".join(missing) + ". Identity review required; lexical matches alone do not establish identity.")
        original_reason = grounded.get("reason", "")
        grounded["reason"] = original_reason + ("\n" if original_reason else "") + explanation
    return grounded
