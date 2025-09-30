PYTHON := python3
PIP := pip
BLACK := black
FLAKE8 := flake8
PYTEST := pytest

PYTHON_FILES := happiness_analysis.py comparison_analysis.py test_happiness.py

.PHONY: all
all: install format lint test

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 pytest pytest-cov

.PHONY: format
format:
	$(BLACK) $(PYTHON_FILES)
	@echo "Code formatting with Black complete"

.PHONY: format-check
format-check:
	$(BLACK) --check $(PYTHON_FILES)

.PHONY: lint
lint:
	$(FLAKE8) --ignore=E501,W503,E203,E226 $(PYTHON_FILES)
	@echo "Linting complete"

.PHONY: test
test:
	$(PYTHON) test_happiness.py
	@echo "Tests complete"

.PHONY: analyze
analyze:
	$(PYTHON) happiness_analysis.py
	@echo "Happiness analysis complete"

.PHONY: compare
compare:
	$(PYTHON) comparison_analysis.py
	@echo "Comparison analysis complete"

.PHONY: clean
clean:
	rm -f *.png
	rm -rf __pycache__
	rm -rf .pytest_cache
	@echo "Cleaned up generated files"

.PHONY: pipeline
pipeline: format lint test analyze compare
	@echo "Full pipeline complete"

.PHONY: ci
ci: format-check lint test
	@echo "CI checks passed"

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install       - Install all dependencies"
	@echo "  format        - Format code with Black"
	@echo "  format-check  - Check formatting without modifying"
	@echo "  lint          - Lint code with flake8"
	@echo "  test          - Run tests"
	@echo "  analyze       - Run main analysis"
	@echo "  compare       - Run comparison analysis"
	@echo "  clean         - Clean generated files"
	@echo "  pipeline      - Run full pipeline"
	@echo "  help          - Show this help message"