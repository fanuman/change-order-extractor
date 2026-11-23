"""Measure extraction accuracy and confidence calibration on the labelled documents.

    python eval/run_eval.py                  # replay cached LLM replies; call the LLM only on a cache miss
    python eval/run_eval.py --refresh        # call the LLM for every document and update the cache
    python eval/run_eval.py --save baseline  # also write eval/results/baseline.json

Each LLM reply is cached in eval/cache/, keyed by prompt version, model and document text. Code changes
(grounding, normalisation, validation, confidence) are measured by replaying the same replies, so a
difference between two runs comes from the code, not from the LLM answering differently. A prompt or
model change misses the cache and calls the LLM.

Run with PYTHONPATH=src (make eval sets it).
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from co_extract.confidence import WEIGHTS
from co_extract.ingest import load
from co_extract.llm import ConfigError, Extractor, LLMConfig, OpenAICompatibleExtractor
from co_extract.pipeline import FIELDS, ChangeOrderPipeline
from co_extract.prompts import PROMPT_VERSION, RawExtraction

EVAL_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = EVAL_DIR / "documents"
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"

EVAL_TODAY = dt.date(2026, 11, 23)  # fixed, so date checks give the same results on any day
DESCRIPTION_SIMILARITY = 0.85  # a description copied slightly differently still counts as correct
CONFIDENCE_BANDS = [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]


# --- Cached LLM replies --------------------------------------------------------------------


class CachedExtractor:
    """Replays saved LLM replies so runs are repeatable and free; calls the real LLM on a cache miss."""

    def __init__(self, live: OpenAICompatibleExtractor, refresh: bool) -> None:
        self.live = live
        self.refresh = refresh
        self.hits = 0

    def extract(self, document_text: str) -> RawExtraction:
        key = f"{PROMPT_VERSION}\n{self.live.config.model}\n{document_text}"
        path = CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
        if path.exists() and not self.refresh:
            self.hits += 1
            return RawExtraction.model_validate_json(path.read_text(encoding="utf-8"))
        raw = self.live.extract(document_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return raw


# --- Scoring one field ---------------------------------------------------------------------


def _plain(text: str) -> str:
    return " ".join(text.split()).casefold()


def outcome(field: str, predicted: Any, expected: Any) -> str:
    """correct, wrong (a different value), missed (null but present) or invented (value but absent)."""
    if expected is None:
        return "correct" if predicted is None else "invented"
    if predicted is None:
        return "missed"
    if field == "amount":
        same = abs(float(predicted) - float(expected)) <= 0.01
    elif field == "description":
        same = SequenceMatcher(None, _plain(predicted), _plain(expected)).ratio() >= DESCRIPTION_SIMILARITY
    elif isinstance(expected, str):
        same = _plain(str(predicted)) == _plain(expected)
    else:
        same = predicted == expected
    return "correct" if same else "wrong"


# --- Running and reporting -----------------------------------------------------------------


def evaluate(pipeline: ChangeOrderPipeline) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every labelled document. Returns one row per field and one row per document."""
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["documents"]
    field_rows, document_rows = [], []
    for name, label in truth.items():
        document = load(DOCUMENTS_DIR / name)
        result = pipeline.run(document).model_dump(mode="json")
        readable = bool(document.text)
        document_rows.append(
            {
                "document": name,
                "readable": readable,
                "needs_review": result["needs_review"],
                "expected_needs_review": label["expect_needs_review"],
                "review_reasons": result["review_reasons"],
            }
        )
        for field, expected in label["expected"].items():
            predicted, confidence = result[field]["value"], result[field]["confidence"]
            field_rows.append(
                {
                    "document": name,
                    "readable": readable,
                    "field": field,
                    "predicted": predicted,
                    "expected": expected,
                    "confidence": confidence,
                    "outcome": outcome(field, predicted, expected),
                }
            )
    return field_rows, document_rows


def summarize(field_rows: list[dict[str, Any]], document_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in field_rows if row["readable"]]
    correct = [row["outcome"] == "correct" for row in rows]
    bands = []
    for low, high in CONFIDENCE_BANDS:
        in_band = [row for row in rows if low <= row["confidence"] < high]
        right = sum(row["outcome"] == "correct" for row in in_band)
        bands.append(
            {
                "band": f"{low:.2f}-{min(high, 1.0):.2f}",
                "fields": len(in_band),
                "correct": right,
                "accuracy": round(right / len(in_band), 3) if in_band else None,
                "mean_confidence": round(sum(r["confidence"] for r in in_band) / len(in_band), 3) if in_band else None,
            }
        )
    return {
        "readable_documents": sum(row["readable"] for row in document_rows),
        "unreadable_documents": sum(not row["readable"] for row in document_rows),
        "fields": len(rows),
        "correct": sum(correct),
        "accuracy": round(sum(correct) / len(rows), 3) if rows else None,
        "outcomes": {
            name: sum(row["outcome"] == name for row in rows) for name in ("correct", "wrong", "missed", "invented")
        },
        "needs_review_matches": sum(row["needs_review"] == row["expected_needs_review"] for row in document_rows),
        "documents": len(document_rows),
        "brier_score": round(
            sum((row["confidence"] - ok) ** 2 for row, ok in zip(rows, correct, strict=True)) / len(rows), 4
        )
        if rows
        else None,
        "calibration": bands,
        "overconfident_errors": [
            row for row in rows if row["outcome"] != "correct" and row["confidence"] >= WEIGHTS.review_threshold
        ],
    }


def print_report(summary: dict[str, Any], field_rows: list[dict[str, Any]], model: str) -> None:
    print(f"prompt {PROMPT_VERSION}, model {model}")
    print(
        f"\nReadable documents: {summary['readable_documents']}  "
        f"(unreadable, reported separately: {summary['unreadable_documents']})"
    )
    print(f"Fields correct: {summary['correct']}/{summary['fields']} ({summary['accuracy']:.1%})")
    print("Outcomes: " + ", ".join(f"{name} {count}" for name, count in summary["outcomes"].items()))
    print(f"needs_review as expected: {summary['needs_review_matches']}/{summary['documents']} documents")

    print("\nBy field                  correct / total")
    for field in FIELDS:
        rows = [r for r in field_rows if r["readable"] and r["field"] == field]
        print(f"  {field:24} {sum(r['outcome'] == 'correct' for r in rows):>3} / {len(rows)}")

    print("\nCalibration: how often fields in each confidence band are right")
    print("  band         fields  correct  accuracy  mean confidence")
    for band in summary["calibration"]:
        accuracy = "-" if band["accuracy"] is None else f"{band['accuracy']:.0%}"
        mean = "-" if band["mean_confidence"] is None else f"{band['mean_confidence']:.2f}"
        print(f"  {band['band']}  {band['fields']:>6}  {band['correct']:>7}  {accuracy:>8}  {mean:>15}")
    print(f"Brier score: {summary['brier_score']} (0 is perfect; always answering 0.5 scores 0.25)")

    errors = summary["overconfident_errors"]
    print(f"\nWrong fields with confidence >= {WEIGHTS.review_threshold} (a reviewer would not look): {len(errors)}")
    for row in errors:
        print(
            f"  {row['document']}  {row['field']}: got {row['predicted']!r}, expected {row['expected']!r} "
            f"({row['outcome']}, confidence {row['confidence']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--refresh", action="store_true", help="call the LLM for every document, updating the cache")
    parser.add_argument("--save", metavar="NAME", help="write the full results to eval/results/NAME.json")
    args = parser.parse_args()

    try:
        live = OpenAICompatibleExtractor(LLMConfig.from_env())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    extractor: Extractor = CachedExtractor(live, refresh=args.refresh)
    field_rows, document_rows = evaluate(ChangeOrderPipeline(extractor, today=EVAL_TODAY))
    summary = summarize(field_rows, document_rows)
    print_report(summary, field_rows, live.config.model)
    print(
        f"\nLLM calls: {live.usage.calls} ({live.usage.prompt_tokens + live.usage.completion_tokens} tokens)",
        file=sys.stderr,
    )

    if args.save:
        if not re.fullmatch(r"[\w.-]+", args.save):
            print("error: --save NAME may only contain letters, digits, '.', '-' and '_'", file=sys.stderr)
            return 2
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{args.save}.json"
        report = {
            "name": args.save,
            "prompt_version": PROMPT_VERSION,
            "model": live.config.model,
            "weights": dataclasses.asdict(WEIGHTS),
            "summary": summary,
            "documents": document_rows,
            "fields": field_rows,
        }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved {path.relative_to(EVAL_DIR.parent)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
