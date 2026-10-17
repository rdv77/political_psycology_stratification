"""Keep single-episode expression separate from recurrence and corpus coverage."""
import re


RUBRIC_POLICY_VERSION = "single-episode-expression-v2"
_FREQUENCY = re.compile(
    r"\b(?:repeat\w*|consisten\w*|systematic\w*|regular\w*|frequen\w*|often|recurr\w*|"
    r"sustain\w*|persist\w*|prominen\w*|multiple\s+(?:episodes|occasions|statements|events)|"
    r"повтор\w*|регуляр\w*|систематич\w*|частот\w*|часто|постоян\w*|неоднократ\w*|"
    r"répét\w*|réguli\w*|fréquen\w*)\b", re.IGNORECASE)


def intensity_rubric_issues(scale):
    """A conservative lexical QA gate, not a complete semantic validator."""
    return [{"field": key, "term": match.group(0)}
            for key in ("anchor0", "anchor3", "anchor5")
            for match in _FREQUENCY.finditer(str(scale.get(key, "")))]


def normalize_discovery(data, context, validator):
    """Quarantine a bad proposal without discarding other usable proposals."""
    accepted, issues = [], []
    for index, scale in enumerate(data["scales"]):
        if not context["config"].get("scales") and intensity_rubric_issues(scale):
            issues.append({"index": index, "code": "frequency_in_intensity_anchor"})
            continue
        try:
            # Joint validation also catches repeated labels and incompatible
            # canonical definitions. Approval is never inferred here.
            validator("describe", {"scales": accepted + [scale]}, context)
        except (ValueError, KeyError, TypeError):
            issues.append({"index": index, "code": "invalid_scale_or_evidence"})
            continue
        accepted.append(scale)
    return {"scales": accepted, "quality_control": {"input_scales": len(data["scales"]),
            "skipped_scales": len(issues), "issues": issues}}
