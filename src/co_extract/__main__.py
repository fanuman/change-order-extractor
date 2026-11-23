"""Command line: extract one change order and print the JSON result to stdout.

    python -m co_extract <path>    # a PDF or text file
    python -m co_extract -         # pasted text, read from stdin

stdout carries only the JSON result. A bad document (empty, scanned, unreadable) still produces
JSON with needs_review set, and exits 0. Only usage errors, such as a missing file or missing
settings, exit 2 with a one-line message on stderr.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

from co_extract.ingest import Document, from_text, load
from co_extract.llm import ConfigError, LLMConfig, OpenAICompatibleExtractor
from co_extract.pipeline import ChangeOrderPipeline

USAGE_ERROR = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="co_extract", description="Extract a change order into validated JSON.")
    parser.add_argument("file", help="path to a PDF or text file, or - to read text from stdin")
    args = parser.parse_args(argv)

    try:
        document = read_document(args.file)
        extractor = OpenAICompatibleExtractor(LLMConfig.from_env())
    except (FileNotFoundError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR

    result = ChangeOrderPipeline(extractor, today=dt.date.today()).run(document)
    print(result.model_dump_json(indent=2))

    usage = extractor.usage
    print(
        f"llm: {usage.calls} call(s), {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens",
        file=sys.stderr,
    )
    return 0


def read_document(file: str) -> Document:
    return from_text(sys.stdin.read()) if file == "-" else load(Path(file))


if __name__ == "__main__":
    sys.exit(main())
