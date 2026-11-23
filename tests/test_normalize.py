"""Normalisation rules as tables: input text -> expected value, and the issue expected (if any).

Each row is one rule or edge case. `issue` is a fragment of the expected issue message, or None
when the value should come back clean.
"""

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest

from co_extract.models import ApprovalStatus
from co_extract.normalize import (
    Normalized,
    normalize_amount,
    normalize_date,
    normalize_identifier,
    normalize_schedule_days,
    normalize_status,
    normalize_text,
)


def check(result: Normalized[Any], value: Any, issue: str | None) -> None:
    assert result.value == value
    if issue is None:
        assert result.issues == ()
    else:
        assert len(result.issues) == 1
        assert issue in result.issues[0]


# --- Amount --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "quote", "value", "issue"),
    [
        # plain increases
        ("12,500.00", None, Decimal("12500.00"), None),
        ("$12,500.00", "Add: $12,500.00", Decimal("12500.00"), None),
        ("USD 4 195.00", None, Decimal("4195.00"), None),
        ("750", None, Decimal("750.00"), None),
        # every way of writing a decrease
        ("(1,250.00)", None, Decimal("-1250.00"), None),
        ("-1,250", None, Decimal("-1250.00"), None),
        ("\N{MINUS SIGN}1,250.00", None, Decimal("-1250.00"), None),
        ("1,250.00-", None, Decimal("-1250.00"), None),
        ("1,250.00 CR", None, Decimal("-1250.00"), None),
        ("1,250.00", "Deduct: 1,250.00", Decimal("-1250.00"), None),
        ("1,250.00", "Credit to owner 1,250.00", Decimal("-1250.00"), None),
        # two decrease signals still mean one decrease
        ("(1,250.00)", "Deduct: (1,250.00)", Decimal("-1250.00"), None),
        # brackets around words are not an accounting negative
        ("1,250.00 (labor)", None, Decimal("1250.00"), None),
        # other number formats
        ("1.250,00", None, Decimal("1250.00"), None),
        ("\N{EURO SIGN} 3.000.000", None, Decimal("3000000.00"), None),
        ("1,25", None, Decimal("1.25"), "decimal mark"),
        # zero
        ("No change", None, Decimal("0.00"), None),
        ("$0.00", None, Decimal("0.00"), None),
        # needs a person
        ("(500.00)", "Add: (500.00)", Decimal("-500.00"), "both as an increase and a decrease"),
        ("1,250.00 plus 300.00", None, Decimal("1250.00"), "more than one number"),
        # brackets that restate a written-out amount are not a negative
        ("($6,300.00)", "Six Thousand Three Hundred Dollars ($6,300.00)", Decimal("6300.00"), None),
        ("($600.00)", "Deduct Six Hundred Dollars ($600.00)", Decimal("-600.00"), None),
        # "CR" only means credit after a number, not in a change request number
        ("$6,300.00", "CR-14 amount: $6,300.00", Decimal("6300.00"), None),
        # form labels that name both directions are neutral
        ("$3,200.00", "ADD/(DEDUCT): $3,200.00", Decimal("3200.00"), None),
        ("($3,200.00)", "ADD/(DEDUCT): ($3,200.00)", Decimal("-3200.00"), None),
        ("$3,200.00", "Increase/(Decrease) in Contract Sum: $3,200.00", Decimal("3200.00"), None),
        # multipliers and ambiguous decimals are read but flagged
        ("$1.2 million", None, Decimal("1200000.00"), "multiplier"),
        ("$15k", None, Decimal("15000.00"), "multiplier"),
        ("1,250.00 M&E", None, Decimal("1250.00"), None),
        ("1.250 EUR", None, Decimal("1.25"), "decimal mark"),
        # unreadable
        ("see attached", None, None, "could not read an amount"),
        ("1,2345", None, None, "could not read the number"),
    ],
)
def test_amount(text: str, quote: str | None, value: Decimal | None, issue: str | None) -> None:
    check(normalize_amount(text, quote), value, issue)


# --- Schedule ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "quote", "value", "issue"),
    [
        ("14 calendar days", None, 14, None),
        ("+14 days", None, 14, None),
        ("Extend 7 days", None, 7, None),
        ("2 weeks", None, 14, None),
        ("0 days", None, 0, None),
        ("No change", None, 0, None),
        ("No impact", None, 0, None),
        # reductions
        ("(5) days", None, -5, None),
        ("-5 days", None, -5, None),
        ("5 days", "Reduce contract time by 5 days", -5, None),
        # brackets that restate a written-out number are not a reduction
        ("fourteen (14) calendar days", None, 14, None),
        ("five (5) days", "Reduce contract time by five (5) days", -5, None),
        # a label that names both directions is neutral
        ("5 days", "Increase/(Decrease) in Contract Time: 5 days", 5, None),
        # converted or assumed, so flagged
        ("1 month", None, 30, "30 days per month"),
        ("10 working days", None, 10, "working days"),
        ("14", None, 14, "no unit"),
        ("N/A", None, 0, "as no change"),
        # not decided or not readable
        ("TBD", None, None, "not determined"),
        ("To be determined", None, None, "not determined"),
        ("2.5 days", None, None, "not a whole number"),
        ("see schedule", None, None, "could not read"),
    ],
)
def test_schedule_days(text: str, quote: str | None, value: int | None, issue: str | None) -> None:
    check(normalize_schedule_days(text, quote), value, issue)


def test_schedule_with_two_durations_uses_the_first_and_says_so() -> None:
    check(normalize_schedule_days("7 days plus 2 weeks"), 7, "more than one duration")


def test_schedule_marked_both_ways_is_flagged() -> None:
    result = normalize_schedule_days("5 days", "Extend contract time; reduce by 5 days")
    assert result.value == -5
    assert "both as an extension and a reduction" in result.issues[0]


# --- Date ----------------------------------------------------------------------------------

MARCH_14 = dt.date(2025, 3, 14)


@pytest.mark.parametrize(
    ("text", "value", "issue"),
    [
        ("2025-03-14", MARCH_14, None),
        ("03/14/2025", MARCH_14, None),
        ("3-14-25", MARCH_14, None),
        ("14.03.2025", MARCH_14, None),  # 14 cannot be a month, so the order is clear
        ("March 14, 2025", MARCH_14, None),
        ("Mar. 14th, 2025", MARCH_14, None),
        ("14 Mar 2025", MARCH_14, None),
        ("14-Mar-2025", MARCH_14, None),
        ("Sept. 3, 2025", dt.date(2025, 9, 3), None),
        ("Date: 03/14/2025", MARCH_14, None),
        ("Order 14, 2025 dated March 2, 2025", dt.date(2025, 3, 2), None),
        ("04/04/2025", dt.date(2025, 4, 4), None),  # same either way
        # ambiguous: both numbers could be the month
        ("03/04/2025", dt.date(2025, 3, 4), "month/day or day/month"),
        # not a date
        ("02/30/2025", None, "not a real calendar date"),
        ("n/a", None, "could not read a date"),
    ],
)
def test_date(text: str, value: dt.date | None, issue: str | None) -> None:
    check(normalize_date(text), value, issue)


# --- Approval status -----------------------------------------------------------------------

APPROVED, PENDING, REJECTED = ApprovalStatus.approved, ApprovalStatus.pending, ApprovalStatus.rejected


@pytest.mark.parametrize(
    ("text", "value", "issue"),
    [
        ("APPROVED", APPROVED, None),
        ("Approved as noted", APPROVED, None),
        ("Accepted by Owner", APPROVED, None),
        ("Executed", APPROVED, None),
        ("Pending", PENDING, None),
        ("Submitted for approval", PENDING, None),
        ("Awaiting owner review", PENDING, None),
        ("Unsigned", PENDING, None),
        ("Rejected", REJECTED, None),
        ("Not approved", REJECTED, None),  # contains "approved" but means the opposite
        # mixed: the most cautious reading wins
        ("Approved by owner, pending architect", PENDING, "mixed"),
        # signatures
        ("/s/ J. Smith 3/15/2025", APPROVED, "inferred from a signature"),
        ("Owner: signed", APPROVED, "inferred from a signature"),
        ("Owner signature: ______", PENDING, "blank signature line"),
        ("Approved by: ________", PENDING, "blank signature line"),  # the label is not the approval
        ("Authorized Signature: ____", PENDING, "blank signature line"),
        ("Accepted: ____", PENDING, "blank signature line"),
        ("Approved by: /s/ J. Alvarez", APPROVED, None),
        # not one of our statuses
        ("VOID", None, "not approved, pending or rejected"),
        ("Withdrawn", None, "not approved, pending or rejected"),
        ("see page 2", None, "could not read an approval status"),
    ],
)
def test_status(text: str, value: ApprovalStatus | None, issue: str | None) -> None:
    check(normalize_status(text), value, issue)


def test_status_reads_the_quote_too() -> None:
    check(normalize_status("J. Smith", "Approved: J. Smith"), APPROVED, None)


# --- Text and identifiers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [("  Riverside   Library\nRenovation ", "Riverside Library Renovation"), ("C-2291", "C-2291")],
)
def test_text_collapses_whitespace(text: str, value: str) -> None:
    check(normalize_text(text), value, None)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("No. 7", "7"),
        ("no 7", "7"),
        ("# C-2291", "C-2291"),
        ("Number: 12", "12"),
        ("CO-007", "CO-007"),
        ("NOVA-2201", "NOVA-2201"),  # "NO" inside an identifier is not a prefix
        ("NO-1234", "NO-1234"),
        ("Number12", "Number12"),
    ],
)
def test_identifier_drops_number_prefixes(text: str, value: str) -> None:
    check(normalize_identifier(text), value, None)


@pytest.mark.parametrize("normalize", [normalize_text, normalize_identifier])
def test_empty_text_is_unreadable(normalize: Any) -> None:
    result = normalize("   ")
    assert result.value is None
    assert len(result.issues) == 1
