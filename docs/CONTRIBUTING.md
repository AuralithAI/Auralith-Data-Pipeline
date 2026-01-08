# Contributing to Auralith Data Pipeline

Thank you for your interest in contributing to Auralith Data Pipeline!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [Ruff](https://beta.ruff.rs/) for linting
- We use [mypy](https://mypy.readthedocs.io/) for type checking

Run all checks:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src/auralith_pipeline --cov-report=html
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and linting is clean
4. Update documentation if needed
5. Submit a pull request

## Adding a New Data Source

To add a new data source:

1. Add entry to `DATASET_REGISTRY` in `sources/data_sources.py`
2. If custom logic is needed, create a new `DataSource` subclass
3. Add tests for the new source
4. Update documentation

## Adding a New Storage Backend

To add a new storage backend:

1. Create a new class inheriting from `StorageBackend`
2. Implement `upload()`, `download()`, and `list_files()` methods
3. Add to `create_storage_backend()` factory function
4. Add tests for the new backend
