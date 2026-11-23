"""Turn text copied from the document into typed values.

Each function takes the text the LLM copied (and, where nearby words carry meaning, the quote it
came from) and returns the typed value plus any issues found. An issue does not always mean the
value is wrong; it means a person should look. Issues become review reasons and lower confidence.

All functions here are pure: no file access, no network, no LLM calls.
"""

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

from co_extract.grounding import PLAIN_PUNCTUATION
from co_extract.models import ApprovalStatus

T = TypeVar("T")


@dataclass(frozen=True)
class Normalized(Generic[T]):
    value: T | None
    issues: tuple[str, ...] = ()


def _unreadable(issue: str) -> Normalized[T]:
    return Normalized(None, (issue,))


def _words(*patterns: str) -> re.Pattern[str]:
    """A case-insensitive pattern matching any of the given whole words or phrases."""
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


# --- Amount --------------------------------------------------------------------------------

_CURRENCY = re.compile(r"\b(?:USD|CAD|AUD|EUR|GBP|US)\b|[$\N{EURO SIGN}\N{POUND SIGN}]", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,. ]*\d|\d")
_DECIMAL_COMMA_ONLY = re.compile(r"\d+,\d{1,2}")  # '1,25': European decimal comma, no thousands mark
_BRACKETED_NUMBER = re.compile(r"\(\s*\d[\d,. ]*\)")  # accounting style for a negative: (1,250.00)
_DECREASE_WORDS = _words(r"deduct\w*", r"decreas\w*", r"credit\w*", r"reduc\w*", r"omi(?:t|ssion)\w*", "minus", "CR")
_INCREASE_WORDS = _words("add", "adds", "added", "addition", "additional", r"increas\w*", "extra")
_ZERO_AMOUNT = re.compile(r"no change|none|no cost|nil|zero", re.IGNORECASE)


def normalize_amount(text: str, quote: str | None = None) -> Normalized[Decimal]:
    """'(1,250.00)' with 'Deduct' nearby -> Decimal('-1250.00'). Increase positive, decrease negative."""
    cleaned = _CURRENCY.sub("", text.translate(PLAIN_PUNCTUATION)).strip()
    if _ZERO_AMOUNT.fullmatch(cleaned):
        return Normalized(Decimal("0.00"))

    numbers = _NUMBER.findall(cleaned)
    if not numbers:
        return _unreadable(f"could not read an amount in {text!r}")
    issues: list[str] = []
    if len(numbers) > 1:
        issues.append(f"amount {text!r} contains more than one number; used the first")
    digits = numbers[0].replace(" ", "")
    number = _parse_number(digits)
    if number is None:
        return _unreadable(f"could not read the number in amount {text!r}")
    if _DECIMAL_COMMA_ONLY.fullmatch(digits):
        issues.append(f"read the comma in amount {text!r} as a decimal mark")

    marked_negative = bool(_BRACKETED_NUMBER.search(cleaned)) or cleaned.startswith("-") or cleaned.endswith("-")
    context = f"{text} {quote or ''}"
    decrease = marked_negative or bool(_DECREASE_WORDS.search(context))
    increase = bool(_INCREASE_WORDS.search(context))
    if decrease and increase:
        issues.append(f"amount {text!r} is marked both as an increase and a decrease; read as a decrease")

    value = -number if decrease else number
    return Normalized(value, tuple(issues))


def _parse_number(digits: str) -> Decimal | None:
    """Read '1,250.00', '1.250,00', '1250' or '1,25' (decimal comma) as a Decimal with 2 places."""
    if "," in digits and "." in digits:
        decimal_mark = "," if digits.rfind(",") > digits.rfind(".") else "."
        thousands_mark = "." if decimal_mark == "," else ","
        digits = digits.replace(thousands_mark, "").replace(decimal_mark, ".")
    elif "," in digits:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", digits):
            digits = digits.replace(",", "")
        elif _DECIMAL_COMMA_ONLY.fullmatch(digits):
            digits = digits.replace(",", ".")
        else:
            return None
    elif digits.count(".") > 1:
        if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", digits):
            return None
        digits = digits.replace(".", "")
    return Decimal(digits).quantize(Decimal("0.01"))


# --- Schedule ------------------------------------------------------------------------------

_DURATION = re.compile(
    r"(?P<number>\(?[-+]?\d+(?:\.\d+)?\)?)\s*(?P<kind>calendar|working|business|work)?\s*(?P<unit>day|week|month)s?\b",
    re.IGNORECASE,
)
_NO_SCHEDULE_CHANGE = _words("no change", "none", "no impact", "no extension", "unchanged", "not affected")
_NOT_APPLICABLE = _words("n/a", "na", "not applicable")
_UNDETERMINED = _words("tbd", "to be determined", "unknown", "pending")
_SHORTER_WORDS = _words(r"reduc\w*", r"decreas\w*", r"shorten\w*", r"accelerat\w*", r"deduct\w*", "earlier")
_LONGER_WORDS = _words(r"extend\w*", "extension", r"increas\w*", "add", "added", "additional", r"delay\w*")
_DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30}


def normalize_schedule_days(text: str, quote: str | None = None) -> Normalized[int]:
    """'14 calendar days' -> 14, '(5) days' -> -5, 'No change' -> 0, 'TBD' -> None."""
    cleaned = text.translate(PLAIN_PUNCTUATION).strip()
    durations = list(_DURATION.finditer(cleaned))

    if not durations:
        if _UNDETERMINED.search(cleaned):
            return _unreadable(f"schedule impact is not determined yet ({text!r})")
        if _NO_SCHEDULE_CHANGE.search(cleaned):
            return Normalized(0)
        if _NOT_APPLICABLE.fullmatch(cleaned):
            return Normalized(0, (f"read schedule impact {text!r} as no change",))
        if re.fullmatch(r"\(?[-+]?\d+\)?", cleaned):
            durations = list(re.finditer(r"(?P<number>\(?[-+]?\d+\)?)", cleaned))
        else:
            return _unreadable(f"could not read a schedule impact in {text!r}")

    issues: list[str] = []
    if len(durations) > 1:
        issues.append(f"schedule impact {text!r} contains more than one duration; used the first")
    first = durations[0]
    unit = (first.groupdict().get("unit") or "").lower()
    kind = (first.groupdict().get("kind") or "").lower()
    if not unit:
        unit = "day"
        issues.append(f"schedule impact {text!r} has no unit; assumed calendar days")
    if unit == "month":
        issues.append(f"schedule impact {text!r} is in months; converted at 30 days per month")
    if kind in ("working", "business", "work"):
        issues.append(f"schedule impact {text!r} is in working days, which may differ from calendar days")

    raw_number = first.group("number")
    magnitude = Decimal(raw_number.strip("()+-"))
    if magnitude != magnitude.to_integral_value():
        return _unreadable(f"schedule impact {text!r} is not a whole number of days")
    days = int(magnitude) * _DAYS_PER_UNIT[unit]

    context = f"{text} {quote or ''}"
    shorter = raw_number.startswith(("(", "-")) or bool(_SHORTER_WORDS.search(context))
    longer = bool(_LONGER_WORDS.search(context))
    if shorter and longer:
        issues.append(f"schedule impact {text!r} is marked both as an extension and a reduction")
    return Normalized(-days if shorter else days, tuple(issues))


# --- Date ----------------------------------------------------------------------------------

_MONTH_NAMES = [
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may", "may"),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sep", "sept"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
]
_MONTHS = {name: number for number, names in enumerate(_MONTH_NAMES, start=1) for name in names}

_ISO_DATE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
_NUMERIC_DATE = re.compile(r"\b(?P<first>\d{1,2})[/.-](?P<second>\d{1,2})[/.-](?P<year>\d{4}|\d{2})\b")
_MONTH_DAY_YEAR = re.compile(r"\b(?P<month>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b")
_DAY_MONTH_YEAR = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?[\s-]+(?P<month>[A-Za-z]{3,9})\.?[\s,-]+(?P<year>\d{4})\b"
)


def normalize_date(text: str) -> Normalized[dt.date]:
    """'03/14/2025', 'March 14, 2025' or '14 Mar 2025' -> date(2025, 3, 14). Ambiguous orders are flagged."""
    cleaned = text.translate(PLAIN_PUNCTUATION)
    issues: list[str] = []

    if match := _ISO_DATE.search(cleaned):
        year, month, day = int(match["year"]), int(match["month"]), int(match["day"])
    elif match := _NUMERIC_DATE.search(cleaned):
        first, second, year = int(match["first"]), int(match["second"]), int(match["year"])
        if year < 100:
            year += 2000
        if first > 12:
            day, month = first, second
        else:
            month, day = first, second
            if second <= 12 and first != second:
                issues.append(f"date {text!r} could be month/day or day/month; read as month/day")
    elif named := _find_named_month_date(cleaned):
        year, month, day = named
    else:
        return _unreadable(f"could not read a date in {text!r}")

    try:
        return Normalized(dt.date(year, month, day), tuple(issues))
    except ValueError:
        return _unreadable(f"date {text!r} is not a real calendar date")


def _find_named_month_date(text: str) -> tuple[int, int, int] | None:
    """Find 'March 14, 2025' or '14 Mar 2025'. Skips capitalised words that are not month names."""
    for pattern in (_MONTH_DAY_YEAR, _DAY_MONTH_YEAR):
        for match in pattern.finditer(text):
            month = _MONTHS.get(match["month"].lower())
            if month is not None:
                return int(match["year"]), month, int(match["day"])
    return None


# --- Approval status -----------------------------------------------------------------------

_NOT_A_STATUS = _words("void", "voided", "cancell?ed", "withdrawn", "superseded")
_REJECTED = _words("not approved", "rejected", "denied", "declined", "disapproved")
_PENDING = _words(
    "pending", "submitted", "awaiting", "under review", "for approval", "proposed", "draft", "unsigned", "not signed"
)
_APPROVED = _words("approved", "accepted", "executed", r"authori[sz]ed")
_SIGNED = re.compile(r"/s/|\bsigned\b", re.IGNORECASE)  # "signature" alone may be an empty line
_BLANK_SIGNATURE_LINE = re.compile(r"_{3,}")


def normalize_status(text: str, quote: str | None = None) -> Normalized[ApprovalStatus]:
    """'APPROVED' -> approved, 'Submitted for approval' -> pending, '/s/ J. Smith' -> approved (flagged)."""
    context = f"{text} {quote or ''}"
    if _NOT_A_STATUS.search(context):
        return _unreadable(f"status {text!r} is not approved, pending or rejected")

    without_rejections = _REJECTED.sub(" ", context)  # so "not approved" does not also count as approved
    found: list[ApprovalStatus] = []  # most cautious first
    if _REJECTED.search(context):
        found.append(ApprovalStatus.rejected)
    if _PENDING.search(without_rejections):
        found.append(ApprovalStatus.pending)
    if _APPROVED.search(without_rejections):
        found.append(ApprovalStatus.approved)

    if len(found) > 1:
        return Normalized(found[0], (f"status {text!r} is mixed; read as the most cautious, {found[0]}",))
    if found:
        return Normalized(found[0])
    if _SIGNED.search(context):
        return Normalized(ApprovalStatus.approved, (f"approval inferred from a signature ({text!r})",))
    if _BLANK_SIGNATURE_LINE.search(context):
        return Normalized(ApprovalStatus.pending, (f"read a blank signature line as not yet approved ({text!r})",))
    return _unreadable(f"could not read an approval status in {text!r}")


# --- Text ----------------------------------------------------------------------------------

_IDENTIFIER_PREFIX = re.compile(r"^(?:no\.?|number|#)\s*:?\s*", re.IGNORECASE)


def normalize_text(text: str) -> Normalized[str]:
    """Collapse whitespace. Used for names and descriptions."""
    value = " ".join(text.split())
    return Normalized(value) if value else _unreadable("value is empty")


def normalize_identifier(text: str) -> Normalized[str]:
    """'No. 7' -> '7', '# C-2291' -> 'C-2291'. Otherwise kept as written."""
    value = _IDENTIFIER_PREFIX.sub("", " ".join(text.split()))
    return Normalized(value) if value else _unreadable("identifier is empty")
