.PHONY: install test lint validate-manifest reproduce paper clean

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests scripts

validate-manifest:
	scpc validate-manifest

reproduce:
	python scripts/reproduce.py

paper: reproduce
	latexmk -pdf -cd paper/main.tex

clean:
	rm -rf results/* paper/generated/* .pytest_cache .ruff_cache
	touch results/README.md paper/generated/.gitkeep
