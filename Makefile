.PHONY: help install install-dev test test-cov lint format type-check docs clean sync

help:
	@echo "Available commands:"
	@echo "  make install      Install the package with uv"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make sync         Sync all dependencies"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code with black and isort"
	@echo "  make type-check   Run type checking with mypy"
	@echo "  make docs         Build documentation"
	@echo "  make clean        Clean up generated files"

install:
	uv sync

install-dev:
	uv sync --group dev

sync:
	uv sync --all-groups

test:
	PYTHONPATH=./crisp-py uv run pytest crisp-py/tests/ -v

test-cov:
	PYTHONPATH=./crisp-py uv run pytest crisp-py/tests/ -v --cov=crisp --cov-report=html --cov-report=term-missing

lint:
	uv run flake8 crisp-py/crisp/ crisp-py/tests/
	uv run black --check crisp-py/crisp/ crisp-py/tests/

format:
	uv run black crisp-py/crisp/ crisp-py/tests/
	uv run isort crisp-py/crisp/ crisp-py/tests/

check:
	uv run mypy crisp-py/crisp/

docs:
	cd docs && uv run make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
