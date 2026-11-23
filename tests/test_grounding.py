import pytest

from co_extract.grounding import Grounding, Match, ground
from co_extract.prompts import Quoted

DOCUMENT = (
    "CHANGE ORDER No. 7\n"
    "Project: Riverside Library Renovation\n"
    "Total change:  Deduct:\n"
    "(1,250.00)\n"
    "Owner\N{RIGHT SINGLE QUOTATION MARK}s request: City of Riverside\n"
    "Schedule: 2025\N{EN DASH}03 start, No change"
)


def ground_quote(text: str | None, quote: str | None, document: str = DOCUMENT) -> Grounding | None:
    return ground(Quoted(text=text, quote=quote), document)


def evidence(grounding: Grounding | None, document: str = DOCUMENT) -> str:
    assert grounding is not None and grounding.start is not None and grounding.end is not None
    return document[grounding.start : grounding.end]


# --- Nothing to ground ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [None, ""])
def test_no_extracted_text_means_nothing_to_ground(text: str | None) -> None:
    assert ground_quote(text, "anything") is None


@pytest.mark.parametrize("quote", [None, "", "   \n "])
def test_text_without_a_quote_has_no_evidence(quote: str | None) -> None:
    grounding = ground_quote("7", quote)
    assert grounding == Grounding(Match.NOT_FOUND, text_in_quote=False)
    assert grounding is not None and not grounding.found


# --- Exact matches -------------------------------------------------------------------------


def test_exact_quote_points_at_its_evidence() -> None:
    grounding = ground_quote("Riverside Library Renovation", "Project: Riverside Library Renovation")
    assert grounding is not None
    assert grounding.match is Match.EXACT
    assert grounding.found
    assert grounding.text_in_quote
    assert evidence(grounding) == "Project: Riverside Library Renovation"


def test_surrounding_whitespace_in_the_quote_is_ignored() -> None:
    grounding = ground_quote("7", "  CHANGE ORDER No. 7\n")
    assert grounding is not None and grounding.match is Match.EXACT
    assert evidence(grounding) == "CHANGE ORDER No. 7"


def test_exact_match_at_the_very_end_of_the_document() -> None:
    grounding = ground_quote("No change", "No change")
    assert grounding is not None and grounding.match is Match.EXACT
    assert grounding.end == len(DOCUMENT)


# --- Normalized matches --------------------------------------------------------------------


def test_quote_across_a_pdf_line_break_maps_back_to_the_original_lines() -> None:
    grounding = ground_quote("(1,250.00)", "Deduct: (1,250.00)")
    assert grounding is not None
    assert grounding.match is Match.NORMALIZED
    assert grounding.text_in_quote
    assert evidence(grounding) == "Deduct:\n(1,250.00)"


@pytest.mark.parametrize(
    ("quote", "original"),
    [
        ("Owner's request: City of Riverside", "Owner\N{RIGHT SINGLE QUOTATION MARK}s request: City of Riverside"),
        ("Schedule: 2025-03 start", "Schedule: 2025\N{EN DASH}03 start"),
        ("change order no. 7", "CHANGE ORDER No. 7"),
        ("Total change: Deduct:", "Total change:  Deduct:"),
    ],
    ids=["curly apostrophe", "en dash", "letter case", "double space"],
)
def test_normalized_match_still_points_at_the_original_text(quote: str, original: str) -> None:
    grounding = ground_quote(quote.split()[-1], quote)
    assert grounding is not None
    assert grounding.match is Match.NORMALIZED
    assert evidence(grounding) == original


# --- Not found -----------------------------------------------------------------------------


def test_invented_quote_is_not_found() -> None:
    grounding = ground_quote("25,000.00", "Add: 25,000.00")
    assert grounding == Grounding(Match.NOT_FOUND, text_in_quote=True)


def test_nothing_is_found_in_an_empty_document() -> None:
    grounding = ground_quote("7", "CHANGE ORDER No. 7", document="")
    assert grounding is not None and grounding.match is Match.NOT_FOUND


# --- Does the quote support the value? -----------------------------------------------------


def test_real_quote_that_does_not_contain_the_value() -> None:
    grounding = ground_quote("(1,250.00)", "CHANGE ORDER No. 7")
    assert grounding is not None
    assert grounding.match is Match.EXACT
    assert not grounding.text_in_quote


def test_value_inside_the_quote_ignoring_spacing_and_case() -> None:
    grounding = ground_quote("city of  RIVERSIDE", "Owner's request: City of Riverside")
    assert grounding is not None and grounding.text_in_quote


# --- The quote supports the value, without false alarms -----------------------------------


def test_currency_symbol_inside_the_quote_is_ignored() -> None:
    document = "This change order:            +$9,850.00"
    grounding = ground_quote("+9,850.00", "This change order:            +$9,850.00", document)
    assert grounding is not None and grounding.text_in_quote


def test_digits_are_not_matched_across_thousands_separators() -> None:
    document = "Add: 12,500.00"
    grounding = ground_quote("1,250", "Add: 12,500.00", document)
    assert grounding is not None and not grounding.text_in_quote


def test_long_value_running_past_its_quote_is_supported_by_the_document() -> None:
    document = "Description of change:\nAdd fire-rated door assembly at stair 2,\nper RFI-044."
    grounding = ground_quote(
        "Add fire-rated door assembly at stair 2, per RFI-044.",
        "Description of change:\nAdd fire-rated door assembly at stair 2,",
        document,
    )
    assert grounding is not None and grounding.text_in_quote


def test_long_value_not_in_the_document_is_not_supported() -> None:
    document = "Description of change:\nAdd fire-rated door assembly at stair 2,\nper RFI-044."
    grounding = ground_quote(
        "Add fire-rated door assembly at stair 2 and stair 3.",
        "Description of change:\nAdd fire-rated door assembly at stair 2,",
        document,
    )
    assert grounding is not None and not grounding.text_in_quote


def test_short_value_outside_its_quote_is_not_supported_even_if_it_appears_elsewhere() -> None:
    grounding = ground_quote("7", "Project: Riverside Library Renovation")  # "7" is in DOCUMENT, but not in the quote
    assert grounding is not None and not grounding.text_in_quote


@pytest.mark.parametrize(
    ("value", "quote", "supported"),
    [
        ("$6,300.00", "Add $16,300.00", False),  # a value is not part of a longer number
        ("300.00", "Add $6,300.00", False),
        ("7", "CHANGE ORDER No. 17", False),
        ("1,250", "Total 1,250,000.00", False),
        ("1,250", "Deduct 1,250.00", True),  # cents may follow
        ("7", "CHANGE ORDER No.7", True),  # letters and a dot before the number are fine
        ("$6,300.00", "Add $6,300.00 per RFI", True),
    ],
)
def test_numbers_are_matched_as_whole_values(value: str, quote: str, supported: bool) -> None:
    grounding = ground_quote(value, quote, document=quote)
    assert grounding is not None and grounding.text_in_quote is supported
