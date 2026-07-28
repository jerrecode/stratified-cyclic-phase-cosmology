.PHONY: install test lint typecheck verify simulate compare data-validate paper clean

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

verify: lint typecheck test data-validate

simulate:
	scpc simulate configs/baseline/scpc_closed.yaml --output results/baseline

compare:
	scpc compare configs/comparison/background_models.yaml --output results/model_comparison

data-validate:
	scpc data validate

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache paper/build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
