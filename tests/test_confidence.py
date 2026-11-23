from collections.abc import Callable
from dataclasses import replace

import pytest

from co_extract.confidence import WEIGHTS, FieldEvidence, Weights, field_confidence, is_low, is_mentioned
from co_extract.grounding import Grounding, Match

EXACT = Grounding(Match.EXACT, text_in_quote=True, start=0, end=18)
NORMALIZED = Grounding(Match.NORMALIZED, text_in_quote=True, start=0, end=18)
NOT_FOUND = Grounding(Match.NOT_FOUND, text_in_quote=True)
QUOTE = "Deduct: (1,250.00)"

CLEAN_EXACT = FieldEvidence(EXACT, has_value=True, quote=QUOTE)


# --- Scores for typical situations ---------------------------------------------------------


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (CLEAN_EXACT, 0.95),
        (FieldEvidence(NORMALIZED, has_value=True, quote=QUOTE), 0.80),
        (replace(CLEAN_EXACT, normalization_issues=("ambiguous date",)), 0.66),
        (replace(CLEAN_EXACT, normalization_issues=("one", "two")), 0.47),
        (replace(CLEAN_EXACT, validation_findings=("sums don't add up",)), 0.38),
        (replace(CLEAN_EXACT, grounding=replace(EXACT, text_in_quote=False)), 0.47),
        (replace(CLEAN_EXACT, quote="No. 7"), 0.81),
        (FieldEvidence(NOT_FOUND, has_value=True, quote="invented"), 0.20),
        (FieldEvidence(EXACT, has_value=False, quote="Date: soon"), 0.20),
        (FieldEvidence(None, has_value=False), 0.90),
        (FieldEvidence(None, has_value=False, mentioned_in_document=True), 0.50),
        (FieldEvidence(None, has_value=False, validation_findings=("no amount was found",)), 0.36),
    ],
    ids=[
        "exact quote",
        "normalized quote",
        "one normalization issue",
        "two normalization issues",
        "validation finding",
        "quote lacks the value",
        "short quote",
        "quote not in document",
        "unreadable value",
        "absent and not mentioned",
        "absent but mentioned",
        "absent with a finding",
    ],
)
def test_confidence(evidence: FieldEvidence, expected: float) -> None:
    assert field_confidence(evidence) == expected


# --- Properties that make the ranking trustworthy before calibration ----------------------

BASES = [
    CLEAN_EXACT,
    FieldEvidence(NORMALIZED, has_value=True, quote=QUOTE),
    FieldEvidence(None, has_value=False),
    FieldEvidence(None, has_value=False, mentioned_in_document=True),
]

PROBLEMS: dict[str, Callable[[FieldEvidence], FieldEvidence]] = {
    "normalization issue": lambda e: replace(e, normalization_issues=(*e.normalization_issues, "issue")),
    "validation finding": lambda e: replace(e, validation_findings=(*e.validation_findings, "finding")),
    "short quote": lambda e: replace(e, quote="No. 7"),
    "quote lacks value": lambda e: replace(e, grounding=e.grounding and replace(e.grounding, text_in_quote=False)),
    "quote not found": lambda e: replace(e, grounding=e.grounding and NOT_FOUND),
    "value unreadable": lambda e: replace(e, has_value=False),
    "label in document": lambda e: replace(e, mentioned_in_document=True),
}


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("problem", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_a_problem_never_raises_confidence(
    base: FieldEvidence, problem: Callable[[FieldEvidence], FieldEvidence]
) -> None:
    assert field_confidence(problem(base)) <= field_confidence(base)


def test_exact_evidence_beats_normalized_evidence() -> None:
    normalized = replace(CLEAN_EXACT, grounding=NORMALIZED)
    assert field_confidence(CLEAN_EXACT) > field_confidence(normalized)


@pytest.mark.parametrize("problems", range(6))
def test_confidence_stays_between_zero_and_one(problems: int) -> None:
    evidence = replace(CLEAN_EXACT, normalization_issues=("x",) * problems, validation_findings=("y",) * problems)
    assert 0 <= field_confidence(evidence) <= 1


def test_weights_can_be_tuned_without_changing_the_logic() -> None:
    stricter = Weights(per_normalization_issue=0.6)
    evidence = replace(CLEAN_EXACT, normalization_issues=("ambiguous date",))
    assert field_confidence(evidence, stricter) == 0.57
    assert field_confidence(evidence) == 0.66  # the default weights are unchanged


# --- Review threshold ----------------------------------------------------------------------


@pytest.mark.parametrize(("confidence", "low"), [(0.95, False), (0.70, False), (0.69, True), (0.0, True)])
def test_review_threshold(confidence: float, low: bool) -> None:
    assert WEIGHTS.review_threshold == 0.7
    assert is_low(confidence) is low


# --- Does the document mention a field? ----------------------------------------------------


@pytest.mark.parametrize(
    ("field", "document", "mentioned"),
    [
        ("contract_number", "Contract No. C-2291", True),
        ("contract_number", "Contract #: C-2291", True),
        ("contract_number", "This contract covers the lobby", False),
        ("change_order_number", "CHANGE ORDER No. 7", True),
        ("change_order_number", "CO #7", True),
        ("change_order_number", "Change order request for tile", False),
        ("requested_by", "Requested by: ________", True),
        ("requested_by", "Initiated by the architect", True),
        ("requested_by", "Prepared by J. Smith", False),
        ("date", "Date: ____", True),
        ("amount", "Total: $1,250.00", True),
        ("schedule_impact_days", "Contract time: No change", True),
        ("approval_status", "Owner signature: ______", True),
        ("project_name", "Riverside Library Renovation", False),
        ("unknown_field", "anything", False),
    ],
)
def test_is_mentioned(field: str, document: str, mentioned: bool) -> None:
    assert is_mentioned(field, document) is mentioned
