"""Run documents through every step and build the output.

    extract -> ground -> normalize -> validate -> score -> build output

ChangeOrderPipeline has one method per step, called in order by run(). Each method applies its
step to all fields by calling the module that owns the rules (grounding, normalize, validate,
confidence), where those rules are tested. FIELDS is the one place that says which LLM field
each output field comes from and which normaliser reads it.

needs_review is true exactly when review_reasons is not empty, and every field below the review
threshold contributes at least one reason, so a reviewer always knows what to check.
"""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from co_extract.confidence import FieldEvidence, field_confidence, is_low, is_mentioned
from co_extract.grounding import Grounding, ground
from co_extract.ingest import Document
from co_extract.llm import ExtractionFailed, Extractor
from co_extract.models import ChangeOrderExtraction
from co_extract.normalize import (
    Normalized,
    normalize_amount,
    normalize_date,
    normalize_identifier,
    normalize_schedule_days,
    normalize_status,
    normalize_text,
)
from co_extract.prompts import ExtractionContext, Quoted, RawExtraction
from co_extract.validate import Finding, Values, validate

Normalizer = Callable[[str, str | None], Normalized[Any]]


def _text_only(normalize: Callable[[str], Normalized[Any]]) -> Normalizer:
    """Adapt a normaliser that reads only the value's text, not the words around it."""
    return lambda text, quote: normalize(text)


# output field -> (the LLM field it comes from, the normaliser that turns its text into a value)
FIELDS: dict[str, tuple[str, Normalizer]] = {
    "project_name": ("project_name", _text_only(normalize_text)),
    "contract_number": ("contract_number", _text_only(normalize_identifier)),
    "change_order_number": ("change_order_number", _text_only(normalize_identifier)),
    "date": ("date", _text_only(normalize_date)),
    "requested_by": ("requested_by", _text_only(normalize_text)),
    "description": ("description", _text_only(normalize_text)),
    "amount": ("amount", normalize_amount),
    "schedule_impact_days": ("schedule_impact", normalize_schedule_days),
    "approval_status": ("approval_status", normalize_status),
}


@dataclass
class FieldResult:
    """What the steps learn about one field, filled in as it moves through the pipeline."""

    quoted: Quoted  # what the LLM returned: text and quote
    grounding: Grounding | None  # None when the LLM returned nothing
    value: Any = None  # the typed value, once normalised
    issues: list[str] = field(default_factory=list)  # from normalisation

    @property
    def is_grounded(self) -> bool:
        return self.grounding is not None and self.grounding.found and self.quoted.text is not None


class ChangeOrderPipeline:
    """Turns change order documents into validated output. Reuse one instance for a batch."""

    def __init__(self, extractor: Extractor, today: dt.date) -> None:
        self.extractor = extractor
        self.today = today  # dates are checked against this, so results don't depend on the clock

    def run(self, document: Document) -> ChangeOrderExtraction:
        """Extract one change order. Never raises for a bad document or a failed LLM call."""
        if not document.text:
            return _unreadable(list(document.warnings))
        try:
            raw = self.extractor.extract(document.text)
        except ExtractionFailed as exc:
            return _unreadable([*document.warnings, f"extraction failed: {exc}"])

        fields = self.ground_fields(raw, document.text)
        self.normalize_fields(fields)
        findings = self.validate_fields(fields, raw.context, document.text)
        scores = self.score_fields(fields, findings, document.text)
        return self.build_output(fields, findings, scores, document)

    # --- Steps, in the order run() calls them ----------------------------------------------

    def ground_fields(self, raw: RawExtraction, document_text: str) -> dict[str, FieldResult]:
        """Look for each field's quote in the document. A quote that isn't there gets no value."""
        fields = {}
        for name, (llm_field, _) in FIELDS.items():
            quoted: Quoted = getattr(raw, llm_field)
            fields[name] = FieldResult(quoted=quoted, grounding=ground(quoted, document_text))
        return fields

    def normalize_fields(self, fields: dict[str, FieldResult]) -> None:
        """Turn each grounded field's text into a typed value, keeping any issues found."""
        for name, result in fields.items():
            if not result.is_grounded:
                continue
            _, normalize = FIELDS[name]
            normalized = normalize(result.quoted.text or "", result.quoted.quote)
            result.value = normalized.value
            result.issues.extend(normalized.issues)

    def validate_fields(
        self, fields: dict[str, FieldResult], context: ExtractionContext, document_text: str
    ) -> list[Finding]:
        """Check ranges and cross-field rules, using the context values the LLM found."""
        line_items = (_grounded_amount(item, document_text, use_quote_words=True) for item in context.line_item_amounts)
        values = Values(
            **{name: result.value for name, result in fields.items()},
            is_change_order=context.is_change_order,
            # Contract sums are totals: words like "deduct" near them describe earlier changes, not their sign.
            contract_sum_before=_grounded_amount(context.contract_sum_before, document_text, use_quote_words=False),
            contract_sum_after=_grounded_amount(context.contract_sum_after, document_text, use_quote_words=False),
            line_item_amounts=tuple(amount for amount in line_items if amount is not None),
        )
        return validate(values, self.today)

    def score_fields(
        self, fields: dict[str, FieldResult], findings: list[Finding], document_text: str
    ) -> dict[str, float]:
        """Confidence for each field, from its grounding, issues and validation findings."""
        # A problem with the whole document lowers trust in every value read from it. Saying a field is
        # absent from such a document is still sound, so nulls are not penalised.
        document_findings = _messages_for(None, findings)
        return {
            name: field_confidence(
                FieldEvidence(
                    grounding=result.grounding,
                    has_value=result.value is not None,
                    quote=result.quoted.quote,
                    normalization_issues=tuple(result.issues),
                    validation_findings=(
                        *_messages_for(name, findings),
                        *(document_findings if result.value is not None else ()),
                    ),
                    mentioned_in_document=is_mentioned(name, document_text),
                )
            )
            for name, result in fields.items()
        }

    def build_output(
        self,
        fields: dict[str, FieldResult],
        findings: list[Finding],
        scores: dict[str, float],
        document: Document,
    ) -> ChangeOrderExtraction:
        """Assemble the output object, with a reason for everything a reviewer should check."""
        reasons = [*document.warnings, *_messages_for(None, findings)]
        for name, result in fields.items():
            mentioned = is_mentioned(name, document.text)
            reasons.extend(_field_reasons(name, result, _messages_for(name, findings), scores[name], mentioned))
        return ChangeOrderExtraction(
            **{
                name: {"value": _json_value(result.value), "confidence": scores[name]}
                for name, result in fields.items()
            },
            needs_review=bool(reasons),
            review_reasons=reasons,
        )


# --- Helpers -------------------------------------------------------------------------------


def _grounded_amount(quoted: Quoted, document_text: str, use_quote_words: bool) -> Decimal | None:
    """A context amount, only if its quote is really in the document."""
    grounding = ground(quoted, document_text)
    if grounding is None or not grounding.found or quoted.text is None:
        return None
    return normalize_amount(quoted.text, quoted.quote if use_quote_words else None).value


def _messages_for(field_name: str | None, findings: list[Finding]) -> tuple[str, ...]:
    """Validation messages about one field, or about the whole document when field_name is None."""
    return tuple(finding.message for finding in findings if finding.field == field_name)


def _field_reasons(
    name: str, result: FieldResult, findings: tuple[str, ...], confidence: float, mentioned: bool
) -> list[str]:
    """Plain-language reasons to review one field. Every low-confidence field gets at least one."""
    reasons = [f"{name}: {message}" for message in (*result.issues, *findings)]
    grounding = result.grounding
    if grounding is not None and not grounding.found:
        reasons.append(f"{name}: quote {result.quoted.quote!r} is not in the document; value discarded")
    elif grounding is not None and not grounding.text_in_quote:
        reasons.append(f"{name}: the quote {result.quoted.quote!r} does not contain the value")
    elif grounding is None and mentioned and is_low(confidence):
        reasons.append(f"{name}: not found, though the document seems to mention it")
    if is_low(confidence) and not reasons:
        reasons.append(f"{name}: low confidence ({confidence:.2f})")
    return reasons


def _json_value(value: Any) -> Any:
    """Money becomes a JSON number only here, at the very end."""
    return float(value) if isinstance(value, Decimal) else value


def _unreadable(reasons: list[str]) -> ChangeOrderExtraction:
    """Every field null with confidence 0: nothing could be read, so nothing is known."""
    empty = {"value": None, "confidence": 0.0}
    return ChangeOrderExtraction(
        **{name: empty for name in FIELDS},
        needs_review=True,
        review_reasons=reasons or ["the document could not be read"],
    )
