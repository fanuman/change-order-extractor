"""Check that each quote the LLM returned really appears in the document.

A value is only as good as its evidence. For every field the LLM returns the extracted text and a
quote it says it copied from the document. We look for that quote in the document:

- EXACT: found character for character.
- NORMALIZED: found after ignoring differences in whitespace, letter case and typographic
  punctuation (curly quotes, dashes), which PDF extraction and LLM copying often change.
- NOT_FOUND: not in the document, so the value has no evidence and must not be trusted.

We also check that the extracted text appears inside its quote, otherwise the quote does not
support it. start/end point at the quote in the original document text, so a reviewer can jump
to the source.
"""

from dataclasses import dataclass
from enum import StrEnum

from co_extract.prompts import Quoted

# Typographic characters and their plain equivalents. Written as code points so the source stays ASCII.
PLAIN_PUNCTUATION = str.maketrans(
    {
        0x2018: "'",  # left single quotation mark
        0x2019: "'",  # right single quotation mark
        0x201C: '"',  # left double quotation mark
        0x201D: '"',  # right double quotation mark
        0x2013: "-",  # en dash
        0x2014: "-",  # em dash
        0x2212: "-",  # minus sign
    }
)


class Match(StrEnum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class Grounding:
    match: Match
    text_in_quote: bool
    start: int | None = None  # where the quote starts in the document text
    end: int | None = None  # where it ends (exclusive), so document[start:end] is the evidence

    @property
    def found(self) -> bool:
        return self.match is not Match.NOT_FOUND


def ground(value: Quoted, document: str) -> Grounding | None:
    """Ground one extracted value. Returns None when the LLM found nothing (text is null)."""
    if not value.text:
        return None
    quote = (value.quote or "").strip()
    if not quote:
        return Grounding(Match.NOT_FOUND, text_in_quote=False)

    text_in_quote = _normalize(value.text) in _normalize(quote)

    start = document.find(quote)
    if start != -1:
        return Grounding(Match.EXACT, text_in_quote, start, start + len(quote))

    span = _find_normalized(quote, document)
    if span is not None:
        return Grounding(Match.NORMALIZED, text_in_quote, *span)

    return Grounding(Match.NOT_FOUND, text_in_quote)


def _find_normalized(quote: str, document: str) -> tuple[int, int] | None:
    """Find the quote ignoring whitespace, case and punctuation style; return its span in the original."""
    needle = _normalize(quote)
    haystack, positions = _normalize_with_positions(document)
    at = haystack.find(needle)
    if not needle or at == -1:
        return None
    return positions[at], positions[at + len(needle) - 1] + 1


def _normalize(text: str) -> str:
    return _normalize_with_positions(text)[0].strip()


def _normalize_with_positions(text: str) -> tuple[str, list[int]]:
    """Normalise text, remembering for each output character the index of the character it came from.

    Runs of whitespace become one space, typographic punctuation becomes plain, letters are casefolded.
    The positions list lets a match in the normalised text be mapped back to the original text.
    """
    chars: list[str] = []
    positions: list[int] = []
    after_space = True  # also drops leading whitespace
    for index, char in enumerate(text):
        if char.isspace():
            if not after_space:
                chars.append(" ")
                positions.append(index)
                after_space = True
            continue
        for plain in char.translate(PLAIN_PUNCTUATION).casefold():
            chars.append(plain)
            positions.append(index)
        after_space = False
    return "".join(chars), positions
