"""Confidence for each field, computed from evidence that code has checked.

We never ask the LLM how sure it is: self-reported confidence is poorly matched to accuracy.
Instead each field starts from how well its quote was grounded in the document, and loses
confidence for every problem the later steps found:

    exact quote          0.95      quote does not contain the value   x 0.5
    normalised quote     0.80      very short quote                   x 0.85
                                   each normalisation issue           x 0.7
                                   each validation finding            x 0.4

A null field's confidence says how sure we are that the field is really absent: high when the
document never mentions it, lower when it has a matching label but the LLM found nothing, and
low when a value was found but had to be thrown away (quote not in the document, unreadable).

These weights are a starting point. The eval measures how often fields at each confidence
level are actually correct, and the weights are tuned until the two agree (calibration).
"""

import re
from dataclasses import dataclass

from co_extract.grounding import Grounding, Match


@dataclass(frozen=True)
class Weights:
    exact_quote: float = 0.95
    normalized_quote: float = 0.80
    quote_lacks_value: float = 0.5
    short_quote: float = 0.85
    per_normalization_issue: float = 0.7
    per_validation_finding: float = 0.4
    absent_and_not_mentioned: float = 0.9
    absent_but_mentioned: float = 0.5
    value_discarded: float = 0.2
    short_quote_length: int = 8
    review_threshold: float = 0.7


WEIGHTS = Weights()


@dataclass(frozen=True)
class FieldEvidence:
    """Everything the earlier steps learned about one field."""

    grounding: Grounding | None  # None when the LLM returned nothing for the field
    has_value: bool  # a typed value survived grounding and normalisation
    quote: str | None = None
    normalization_issues: tuple[str, ...] = ()
    validation_findings: tuple[str, ...] = ()
    mentioned_in_document: bool = False  # the document has a label for this field


def field_confidence(evidence: FieldEvidence, weights: Weights = WEIGHTS) -> float:
    """A score from 0 to 1, rounded to two decimals."""
    if evidence.grounding is None:
        score = weights.absent_but_mentioned if evidence.mentioned_in_document else weights.absent_and_not_mentioned
    elif not evidence.grounding.found or not evidence.has_value:
        score = weights.value_discarded
    else:
        score = weights.exact_quote if evidence.grounding.match is Match.EXACT else weights.normalized_quote
        if not evidence.grounding.text_in_quote:
            score *= weights.quote_lacks_value
        if len((evidence.quote or "").strip()) < weights.short_quote_length:
            score *= weights.short_quote
        score *= weights.per_normalization_issue ** len(evidence.normalization_issues)

    score *= weights.per_validation_finding ** len(evidence.validation_findings)
    return round(score, 2)


def is_low(confidence: float, weights: Weights = WEIGHTS) -> bool:
    return confidence < weights.review_threshold


# Labels that show a document talks about a field. Used only to judge how sure "absent" is.
_FIELD_LABELS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "project_name": r"\bproject\b|\bjob\b",
        "contract_number": r"\bcontract\s*(?:no\b|number|#|:)",
        "change_order_number": r"\bchange\s*order\s*(?:no\b|number|#)|\bC\.?O\.?\s*(?:no\b|#)",
        "date": r"\bdate[d]?\b",
        "requested_by": r"\brequested\s+by\b|\binitiated\s+by\b|\brequestor\b",
        "description": r"\bdescription\b|\bscope\b",
        "amount": r"\bamount\b|\btotal\b|\bcost\b|\bprice\b|\$",
        "schedule_impact_days": r"\bcontract\s+time\b|\bschedule\b|\bcompletion\s+date\b|\bdays\b",
        # approval words, or a signature block: a role followed by a signature or a blank line
        "approval_status": r"\bapprov|\bstatus\b|\bsignature\b|\bsigned\b"
        r"|\b(?:owner|architect|contractor|engineer)\s*:?\s*(?:/s/|_{3,})",
    }.items()
}


def is_mentioned(field: str, document: str) -> bool:
    """Does the document have a label for this field, such as 'Contract No.' or 'Requested by'?"""
    pattern = _FIELD_LABELS.get(field)
    return bool(pattern and pattern.search(document))
