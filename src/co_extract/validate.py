"""Check that the extracted values make sense, beyond their types.

Normalisation looked at each field on its own. Validation looks at ranges (is this value
plausible?) and at fields against each other (do the contract sums agree with the amount?).
A value can be read perfectly and still be the wrong value, for example the new contract sum
picked up instead of this change order's amount; only cross-field rules catch that.

Each rule is a small function. validate() runs every rule in RULES. To add a rule, write one
function and add it to the list. All rules are pure: no file access, no network, no LLM calls.
"""

import datetime as dt
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from decimal import Decimal

from co_extract.models import ApprovalStatus

CENT = Decimal("0.01")  # money is compared to the cent
EARLIEST_PLAUSIBLE_DATE = dt.date(1990, 1, 1)
MAX_PLAUSIBLE_AMOUNT = Decimal("100000000")  # 100 million
MAX_PLAUSIBLE_SCHEDULE_DAYS = 730  # two years either way


@dataclass(frozen=True)
class Values:
    """Typed values for one change order, after normalisation. Context fields are only for checks."""

    project_name: str | None = None
    contract_number: str | None = None
    change_order_number: str | None = None
    date: dt.date | None = None
    requested_by: str | None = None
    description: str | None = None
    amount: Decimal | None = None
    schedule_impact_days: int | None = None
    approval_status: ApprovalStatus | None = None
    is_change_order: bool = True
    contract_sum_before: Decimal | None = None
    contract_sum_after: Decimal | None = None
    line_item_amounts: tuple[Decimal, ...] = ()


@dataclass(frozen=True)
class Finding:
    """One failed check. `field` is the output field to look at, or None for the whole document."""

    field: str | None
    message: str


Rule = Callable[[Values, dt.date], Iterator[Finding]]


def _money(amount: Decimal) -> str:
    return f"{amount:,.2f}"


def validate(values: Values, today: dt.date) -> list[Finding]:
    """Run every rule. `today` is passed in so results don't depend on when the code runs."""
    return [finding for rule in RULES for finding in rule(values, today)]


# --- Document ------------------------------------------------------------------------------


def looks_like_a_change_order(values: Values, today: dt.date) -> Iterator[Finding]:
    if not values.is_change_order:
        yield Finding(None, "the document does not look like a construction change order")


def key_fields_are_present(values: Values, today: dt.date) -> Iterator[Finding]:
    """A change order without an amount or a number can't be used, even if 'absent' is certain."""
    if values.amount is None:
        yield Finding("amount", "no amount was found")
    if values.change_order_number is None:
        yield Finding("change_order_number", "no change order number was found")


# --- Ranges --------------------------------------------------------------------------------


def date_is_plausible(values: Values, today: dt.date) -> Iterator[Finding]:
    if values.date is None:
        return
    if values.date > today:
        yield Finding("date", f"date {values.date.isoformat()} is in the future")
    elif values.date < EARLIEST_PLAUSIBLE_DATE:
        yield Finding("date", f"date {values.date.isoformat()} is before {EARLIEST_PLAUSIBLE_DATE.year}")


def amount_is_plausible(values: Values, today: dt.date) -> Iterator[Finding]:
    if values.amount is not None and abs(values.amount) > MAX_PLAUSIBLE_AMOUNT:
        yield Finding("amount", f"amount {_money(values.amount)} is larger than {MAX_PLAUSIBLE_AMOUNT:,}")


def schedule_is_plausible(values: Values, today: dt.date) -> Iterator[Finding]:
    days = values.schedule_impact_days
    if days is not None and abs(days) > MAX_PLAUSIBLE_SCHEDULE_DAYS:
        yield Finding("schedule_impact_days", f"schedule impact of {days} days is more than two years")


def contract_sum_after_is_not_negative(values: Values, today: dt.date) -> Iterator[Finding]:
    if values.contract_sum_after is not None and values.contract_sum_after < 0:
        yield Finding("amount", f"the new contract sum {_money(values.contract_sum_after)} is negative")


# --- Fields against each other -------------------------------------------------------------


def contract_sums_add_up(values: Values, today: dt.date) -> Iterator[Finding]:
    """Contract sum before + amount = contract sum after, when the document states all three."""
    before, amount, after = values.contract_sum_before, values.amount, values.contract_sum_after
    if before is None or amount is None or after is None:
        return
    if abs(before + amount - after) <= CENT:
        return
    if abs(before - amount - after) <= CENT:
        yield Finding(
            "amount",
            f"the sign of the amount looks reversed: {_money(before)} to {_money(after)} "
            f"is a change of {_money(after - before)}, not {_money(amount)}",
        )
    else:
        yield Finding(
            "amount",
            f"contract sums don't add up: {_money(before)} + {_money(amount)} = {_money(before + amount)}, "
            f"but the document says {_money(after)}",
        )


def line_items_add_up(values: Values, today: dt.date) -> Iterator[Finding]:
    """Line items total the amount. Items are often listed unsigned under a 'Deduct' heading, so the
    total may match the amount's size rather than its sign."""
    if not values.line_item_amounts or values.amount is None:
        return
    total = sum(values.line_item_amounts, Decimal("0"))
    if abs(total - values.amount) <= CENT or abs(total - abs(values.amount)) <= CENT:
        return
    yield Finding("amount", f"line items add up to {_money(total)}, but the amount is {_money(values.amount)}")


RULES: list[Rule] = [
    looks_like_a_change_order,
    key_fields_are_present,
    date_is_plausible,
    amount_is_plausible,
    schedule_is_plausible,
    contract_sum_after_is_not_negative,
    contract_sums_add_up,
    line_items_add_up,
]
