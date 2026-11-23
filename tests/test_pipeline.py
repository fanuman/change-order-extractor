"""The pipeline end to end, with a fake extractor standing in for the LLM."""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from co_extract.confidence import WEIGHTS
from co_extract.ingest import Document, from_text
from co_extract.llm import ExtractionFailed
from co_extract.models import ChangeOrderExtraction
from co_extract.pipeline import FIELDS, ChangeOrderPipeline
from co_extract.prompts import RawExtraction

TODAY = dt.date(2025, 6, 1)
SCHEMA = json.loads((Path(__file__).parents[1] / "schema" / "change_order.schema.json").read_text())

DOCUMENT = from_text(
    """CHANGE ORDER No. 7
Project: Riverside Library Renovation   Contract No. C-2291
Date: 03/14/2025   Requested by: City of Riverside (Owner)
Description: Replace specified oak flooring with porcelain tile in the lobby.
Deduct: (1,250.00)    Contract time: No change
Original contract sum: 250,000.00   New contract sum: 248,750.00
Status: APPROVED"""
)

NULL = {"text": None, "quote": None}


def quoted(text: str, quote: str) -> dict[str, str]:
    return {"text": text, "quote": quote}


CLEAN_REPLY: dict[str, Any] = {
    "project_name": quoted("Riverside Library Renovation", "Project: Riverside Library Renovation"),
    "contract_number": quoted("C-2291", "Contract No. C-2291"),
    "change_order_number": quoted("7", "CHANGE ORDER No. 7"),
    "date": quoted("03/14/2025", "Date: 03/14/2025"),
    "requested_by": quoted("City of Riverside (Owner)", "Requested by: City of Riverside (Owner)"),
    "description": quoted(
        "Replace specified oak flooring with porcelain tile in the lobby.",
        "Description: Replace specified oak flooring with porcelain tile in the lobby.",
    ),
    "amount": quoted("(1,250.00)", "Deduct: (1,250.00)"),
    "schedule_impact": quoted("No change", "Contract time: No change"),
    "approval_status": quoted("APPROVED", "Status: APPROVED"),
    "context": {
        "is_change_order": True,
        "contract_sum_before": quoted("250,000.00", "Original contract sum: 250,000.00"),
        "contract_sum_after": quoted("248,750.00", "New contract sum: 248,750.00"),
        "line_item_amounts": [],
    },
}


class FakeExtractor:
    """Returns a fixed reply (or raises), and counts calls."""

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply
        self.calls = 0

    def extract(self, document_text: str) -> RawExtraction:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return RawExtraction.model_validate(self.reply)


def with_changes(**changes: Any) -> dict[str, Any]:
    reply = json.loads(json.dumps(CLEAN_REPLY))
    for key, value in changes.items():
        if key in reply["context"]:
            reply["context"][key] = value
        else:
            reply[key] = value
    return reply


def run(reply: dict[str, Any] | Exception, document: Document = DOCUMENT) -> dict[str, Any]:
    result = ChangeOrderPipeline(FakeExtractor(reply), TODAY).run(document)
    output = result.model_dump(mode="json")
    jsonschema.validate(output, SCHEMA)  # every output must match the published schema
    return output


def assert_reasons_explain_review(output: dict[str, Any]) -> None:
    """needs_review is true exactly when there are reasons, and every low-confidence field is explained."""
    reasons = output["review_reasons"]
    assert output["needs_review"] is bool(reasons)
    document_level = [reason for reason in reasons if not any(reason.startswith(f"{name}:") for name in FIELDS)]
    for name in FIELDS:
        if output[name]["confidence"] < WEIGHTS.review_threshold:
            assert document_level or any(reason.startswith(f"{name}:") for reason in reasons), name


# --- Wiring --------------------------------------------------------------------------------


def test_fields_table_covers_every_output_field_and_a_real_llm_field() -> None:
    output_fields = set(ChangeOrderExtraction.model_fields) - {"needs_review", "review_reasons"}
    assert set(FIELDS) == output_fields
    for llm_field, _ in FIELDS.values():
        assert llm_field in RawExtraction.model_fields


# --- Whole documents -----------------------------------------------------------------------


def test_clean_document() -> None:
    output = run(CLEAN_REPLY)
    assert output["amount"] == {"value": -1250.0, "confidence": 0.95}
    assert output["date"]["value"] == "2025-03-14"
    assert output["schedule_impact_days"]["value"] == 0
    assert output["approval_status"]["value"] == "approved"
    assert all(output[name]["confidence"] == 0.95 for name in FIELDS)
    assert output["needs_review"] is False
    assert output["review_reasons"] == []


def test_empty_document_does_not_call_the_llm() -> None:
    extractor = FakeExtractor(CLEAN_REPLY)
    document = Document(text="", source_type="pdf", warnings=("PDF has little or no text layer",))
    output = ChangeOrderPipeline(extractor, TODAY).run(document).model_dump(mode="json")
    jsonschema.validate(output, SCHEMA)
    assert extractor.calls == 0
    assert output["review_reasons"] == ["PDF has little or no text layer"]
    assert all(output[name] == {"value": None, "confidence": 0.0} for name in FIELDS)


def test_llm_failure_gives_nulls_and_a_reason() -> None:
    output = run(ExtractionFailed("model 'x' not found; check LLM_MODEL"))
    assert output["needs_review"] is True
    assert output["review_reasons"] == ["extraction failed: model 'x' not found; check LLM_MODEL"]
    assert all(output[name]["value"] is None for name in FIELDS)


def test_invented_quote_is_discarded() -> None:
    output = run(with_changes(contract_number=quoted("C-1001", "Contract C-1001")))
    assert output["contract_number"]["value"] is None
    assert output["contract_number"]["confidence"] < WEIGHTS.review_threshold
    assert (
        "contract_number: quote 'Contract C-1001' is not in the document; value discarded" in output["review_reasons"]
    )
    assert_reasons_explain_review(output)


def test_value_its_quote_does_not_contain_is_discarded() -> None:
    output = run(with_changes(date=quoted("11/02/2025", "Date: 03/14/2025")))
    assert output["date"] == {"value": None, "confidence": 0.2}
    assert "date: the quote 'Date: 03/14/2025' does not contain the value; value discarded" in output["review_reasons"]


def test_not_a_change_order_lowers_every_value_but_not_the_nulls() -> None:
    reply = with_changes(is_change_order=False, project_name=NULL)
    output = run(reply)
    assert "the document does not look like a construction change order" in output["review_reasons"]
    with_values = [name for name in FIELDS if output[name]["value"] is not None]
    assert len(with_values) == 8
    assert all(output[name]["confidence"] < WEIGHTS.review_threshold for name in with_values)
    # "absent" is still a sound claim about such a document, so the null keeps its usual confidence
    assert output["project_name"] == run(with_changes(project_name=NULL))["project_name"]
    assert_reasons_explain_review(output)


def test_deduct_in_the_quote_makes_the_amount_negative() -> None:
    output = run(with_changes(amount=quoted("1,250.00", "Deduct: (1,250.00)")))  # brackets lost in the copied text
    assert output["amount"]["value"] == -1250.0
    assert output["needs_review"] is False


def test_sums_that_disagree_flag_the_amount() -> None:
    output = run(with_changes(contract_sum_after=quoted("250,000.00", "Original contract sum: 250,000.00")))
    assert output["amount"]["confidence"] < WEIGHTS.review_threshold
    assert any(reason.startswith("amount: contract sums don't add up") for reason in output["review_reasons"])


def test_words_near_a_contract_sum_do_not_flip_its_sign() -> None:
    document = from_text(DOCUMENT.text + "\nPrevious sum (after deduct CO #3): 250,000.00")
    reply = with_changes(contract_sum_before=quoted("250,000.00", "Previous sum (after deduct CO #3): 250,000.00"))
    output = run(reply, document)
    assert output["needs_review"] is False  # 250,000.00 - 1,250.00 = 248,750.00, so the sums agree


def test_context_values_without_evidence_are_ignored() -> None:
    reply = with_changes(contract_sum_after=quoted("999.00", "New contract sum: 999.00"))  # not in the document
    output = run(reply)
    assert output["needs_review"] is False  # the invented sum is not used for the cross-check


@pytest.mark.parametrize(
    "reply",
    [
        CLEAN_REPLY,
        with_changes(date=quoted("03/04/2025", "Date: 03/14/2025")),
        with_changes(requested_by=NULL),
        with_changes(approval_status=quoted("Void", "Status: APPROVED")),
        with_changes(is_change_order=False),
    ],
    ids=["clean", "quote lacks value", "absent but mentioned", "unreadable status", "not a change order"],
)
def test_every_low_confidence_field_is_explained(reply: dict[str, Any]) -> None:
    assert_reasons_explain_review(run(reply))
