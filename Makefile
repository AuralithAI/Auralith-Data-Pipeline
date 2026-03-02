.PHONY: setup install dev test lint format clean build docs release bump

# One-shot setup (venv + all dependencies)
setup:
	chmod +x setup.sh && ./setup.sh

# Install package
install:
pip install -e .

# Install with development dependencies
dev:
pip install -e ".[dev]"
pre-commit install

# Install all dependencies
all:
pip install -e ".[all]"

# Run tests
test:
pytest tests/ -v

# Run tests with coverage
test-cov:
pytest tests/ -v --cov=src/auralith_pipeline --cov-report=html --cov-report=term

# Lint code
lint:
ruff check src/ tests/
mypy src/ --ignore-missing-imports

# Format code
format:
black src/ tests/
ruff check src/ tests/ --fix

# Clean build artifacts
clean:
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf .pytest_cache/
rm -rf .mypy_cache/
rm -rf .coverage
rm -rf htmlcov/
find . -type d -name __pycache__ -exec rm -rf {} +

# Build package
build: clean
	python -m build
	twine check dist/*

# Bump version and release (usage: make release V=2.1.0)
release:
	@test -n "$(V)" || (echo "Usage: make release V=2.1.0" && exit 1)
	chmod +x scripts/bump-version.sh && ./scripts/bump-version.sh $(V) --push

# Bump version without pushing (usage: make bump V=2.1.0)
bump:
	@test -n "$(V)" || (echo "Usage: make bump V=2.1.0" && exit 1)
	chmod +x scripts/bump-version.sh && ./scripts/bump-version.sh $(V)

# Collect Wikipedia sample
collect-sample:
python -m auralith_pipeline.cli collect --dataset wikipedia --max-samples 1000 --output ./data/sample

# Run the CLI
cli:
python -m auralith_pipeline.cli --help
