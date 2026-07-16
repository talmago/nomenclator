.PHONY: install lock format lint typecheck test check clean

install:
	poetry install

lock:
	poetry lock

format:
	poetry run ruff format .
	poetry run ruff check --fix .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

test:
	poetry run pytest tests/unit -m "not integration"

integration:
	poetry run pytest tests/integration

check: lint typecheck

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
