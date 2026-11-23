import datetime as dt
import inspect
from dataclasses import replace
from decimal import Decimal

import pytest

from co_extract import validate as validate_module
from co_extract.validate import RULES, Finding, Values, validate

TODAY = dt.date(2025, 6, 1)

# A change order that passes every rule. Each test changes one thing.
CLEAN = Values(
    project_name="Riverside Library Renovation",
    contract_number="C-2291",
    change_order_number="7",
    date=dt.date(2025, 3, 14),
    amount=Decimal("-1250.00"),
    schedule_impact_days=0,
    contract_sum_before=Decimal("250000.00"),
    contract_sum_after=Decimal("248750.00"),
)


def findings(values: Values) -> list[tuple[str | None, str]]:
    return [(finding.field, finding.message) for finding in validate(values, TODAY)]


def only_finding(values: Values) -> Finding:
    results = validate(values, TODAY)
    assert len(results) == 1, results
    return results[0]


def test_clean_change_order_has_no_findings() -> None:
    assert findings(CLEAN) == []


def test_every_rule_function_is_registered() -> None:
    """A rule that is written but not added to RULES would silently never run."""
    rule_functions = [
        function
        for name, function in inspect.getmembers(validate_module, inspect.isfunction)
        if function.__module__ == validate_module.__name__ and name not in ("validate", "_money")
    ]
    assert sorted(rule_functions, key=lambda f: f.__name__) == sorted(RULES, key=lambda f: f.__name__)


# --- Document ------------------------------------------------------------------------------


def test_document_that_is_not_a_change_order() -> None:
    finding = only_finding(replace(CLEAN, is_change_order=False))
    assert finding.field is None
    assert "does not look like a construction change order" in finding.message


def test_missing_key_fields_are_reported_per_field() -> None:
    assert findings(Values()) == [
        ("amount", "no amount was found"),
        ("change_order_number", "no change order number was found"),
    ]


def test_other_missing_fields_are_not_findings() -> None:
    sparse = Values(change_order_number="7", amount=Decimal("100.00"))
    assert findings(sparse) == []


# --- Date ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "problem"),
    [
        (TODAY, None),
        (dt.date(1990, 1, 1), None),
        (TODAY + dt.timedelta(days=1), "is in the future"),
        (dt.date(1989, 12, 31), "is before 1990"),
    ],
)
def test_date_range(date: dt.date, problem: str | None) -> None:
    results = findings(replace(CLEAN, date=date))
    if problem is None:
        assert results == []
    else:
        assert len(results) == 1 and results[0][0] == "date" and problem in results[0][1]


# --- Amount and schedule ranges ------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "flagged"),
    [
        (Decimal("100000000.00"), False),
        (Decimal("-100000000.00"), False),
        (Decimal("100000000.01"), True),
        (Decimal("-100000000.01"), True),
    ],
)
def test_amount_limit(amount: Decimal, flagged: bool) -> None:
    values = Values(change_order_number="7", amount=amount)  # no contract sums, so only the range rule applies
    assert bool(findings(values)) is flagged


@pytest.mark.parametrize(("days", "flagged"), [(730, False), (-730, False), (731, True), (-731, True)])
def test_schedule_limit(days: int, flagged: bool) -> None:
    results = findings(replace(CLEAN, schedule_impact_days=days))
    assert bool(results) is flagged
    if flagged:
        assert results[0][0] == "schedule_impact_days"


def test_negative_new_contract_sum() -> None:
    values = replace(CLEAN, contract_sum_before=Decimal("1000.00"), contract_sum_after=Decimal("-250.00"))
    assert ("amount", "the new contract sum -250.00 is negative") in findings(values)


# --- Contract sums -------------------------------------------------------------------------


def test_sums_within_one_cent_pass() -> None:
    assert findings(replace(CLEAN, contract_sum_after=Decimal("248750.01"))) == []


def test_sums_off_by_more_than_a_cent_fail() -> None:
    finding = only_finding(replace(CLEAN, contract_sum_after=Decimal("248750.02")))
    assert finding.field == "amount"
    assert "contract sums don't add up" in finding.message


def test_reversed_sign_is_named() -> None:
    finding = only_finding(replace(CLEAN, amount=Decimal("1250.00")))
    assert finding.message == (
        "the sign of the amount looks reversed: 250,000.00 to 248,750.00 is a change of -1,250.00, not 1,250.00"
    )


def test_new_contract_sum_taken_as_the_amount_is_caught() -> None:
    finding = only_finding(replace(CLEAN, amount=Decimal("248750.00")))
    assert "250,000.00 + 248,750.00 = 498,750.00, but the document says 248,750.00" in finding.message


@pytest.mark.parametrize("missing", ["contract_sum_before", "contract_sum_after"])
def test_sums_are_only_checked_when_both_are_stated(missing: str) -> None:
    assert findings(replace(CLEAN, amount=Decimal("999.00"), **{missing: None})) == []


# --- Line items ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "items",
    [
        (Decimal("-1000.00"), Decimal("-250.00")),  # signed like the amount
        (Decimal("1000.00"), Decimal("250.00")),  # listed unsigned under a "Deduct" heading
    ],
    ids=["signed", "unsigned"],
)
def test_line_items_that_total_the_amount_pass(items: tuple[Decimal, ...]) -> None:
    assert findings(replace(CLEAN, line_item_amounts=items)) == []


def test_line_items_that_do_not_total_the_amount_fail() -> None:
    finding = only_finding(replace(CLEAN, line_item_amounts=(Decimal("1000.00"), Decimal("200.00"))))
    assert finding == Finding("amount", "line items add up to 1,200.00, but the amount is -1,250.00")


def test_line_items_are_not_checked_without_an_amount() -> None:
    values = replace(CLEAN, amount=None, contract_sum_before=None, line_item_amounts=(Decimal("5.00"),))
    assert findings(values) == [("amount", "no amount was found")]
