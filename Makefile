# Common commands. Run `make` or `make help` to list them.

PYTHON ?= python
export PYTHONPATH := src

.DEFAULT_GOAL := help
.PHONY: help check lint format format-check typecheck test schema schema-check extract outputs eval

help:
	@echo "make check                 lint, format check, type check, tests, schema up to date"
	@echo "make test                  run the tests (no network)"
	@echo "make format                reformat the code"
	@echo "make extract FILE=<path>   extract one document, JSON on stdout"
	@echo "make outputs               extract every file in samples/ into outputs/"
	@echo "make schema                regenerate schema/change_order.schema.json"
	@echo "make eval                  accuracy and calibration on eval/ (calls the LLM)"

check: lint format-check typecheck test schema-check

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy

test:
	$(PYTHON) -m pytest

schema:
	$(PYTHON) scripts/export_schema.py

schema-check:
	$(PYTHON) scripts/export_schema.py --check

extract:
	@test -n "$(FILE)" || { echo "usage: make extract FILE=<path>" >&2; exit 2; }
	@$(PYTHON) -m co_extract "$(FILE)"

outputs:
	@mkdir -p outputs
	@for f in samples/*; do \
	    name=$$(basename "$$f"); \
	    echo "extracting $$f" >&2; \
	    $(PYTHON) -m co_extract "$$f" > "outputs/$${name%.*}.json" || exit 1; \
	done

eval:
	$(PYTHON) eval/run_eval.py
