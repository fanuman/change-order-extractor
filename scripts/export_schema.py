"""Write the JSON Schema for the output to schema/change_order.schema.json.

python scripts/export_schema.py          # write the file
python scripts/export_schema.py --check  # exit 1 if the file is missing or out of date
"""

import argparse
import json
import sys
from pathlib import Path

from co_extract.models import ChangeOrderExtraction

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "change_order.schema.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def build_schema() -> str:
    """The schema as text. Same models in, same text out, so --check can compare exactly."""
    schema = ChangeOrderExtraction.model_json_schema(mode="serialization")
    schema = {"$schema": SCHEMA_DIALECT, **schema}
    return json.dumps(schema, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or check the output JSON Schema.")
    parser.add_argument("--check", action="store_true", help="fail if the schema file is missing or out of date")
    args = parser.parse_args()

    expected = build_schema()

    if args.check:
        actual = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else None
        if actual != expected:
            print(f"{SCHEMA_PATH.name} is missing or out of date. Run: make schema", file=sys.stderr)
            return 1
        print(f"{SCHEMA_PATH.name} is up to date", file=sys.stderr)
        return 0

    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(expected, encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
