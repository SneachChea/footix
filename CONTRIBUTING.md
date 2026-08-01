# Contributing to Footix

Thank you for your interest in contributing to Footix! This document provides guidelines for submitting contributions.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/SneachChea/footix.git
   cd footix
   ```

2. Install the project with development dependencies:
   ```bash
   uv sync --all-groups
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Development Workflow

### Linting and Formatting

The project uses:
- **Ruff** for code formatting and linting
- **MyPy** for type checking

Run the entire pre-commit pipeline:
```bash
uv run pre-commit run --all-files
```

Or run individual tools:
```bash
uv run ruff check footix/    # Lint code
uv run ruff format footix/   # Format code
uv run mypy footix           # Type-checking
```

### Testing

Run the test suite with coverage:
```bash
uv run pytest -v --cov=footix
```

This includes a strict Sphinx documentation build test.

### Building Documentation

Build the documentation locally:
```bash
uv run sphinx-build -b html -W docs/source docs/build/html
```

Then open `docs/build/html/index.html` in your browser.

Check for broken links:
```bash
uv run sphinx-build -b linkcheck docs/source docs/build/linkcheck
```

## Docstring Standards

Footix uses **Google-style docstrings** for all public modules, classes, and functions. This ensures consistency and enables proper Sphinx autodoc rendering.

### Google-Style Docstring Format

#### Module Docstring

```python
"""Short one-line module summary.

Extended description (if needed) explaining the module's purpose,
key classes, and main functionality.

Example:
    Basic usage example of the module::

        from footix.module import SomeClass
        obj = SomeClass()
        result = obj.method()
"""
```

#### Class Docstring

```python
class MyModel:
    """Short one-line class summary.

    Extended description explaining the class purpose and key behavior.

    Attributes:
        param1 (str): Description of param1.
        param2 (int): Description of param2.

    Example:
        Basic usage::

            model = MyModel(param1="value", param2=42)
            output = model.fit(data)
    """

    def __init__(self, param1: str, param2: int) -> None:
        """Initialize MyModel.

        Args:
            param1: Description of param1.
            param2: Description of param2.
        """
        self.param1 = param1
        self.param2 = param2
```

#### Function Docstring

```python
def compute_metric(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute a custom metric between predictions and targets.

    Extended description (if needed) explaining the metric calculation,
    its use cases, or special behavior.

    Args:
        predictions: Array of predicted values. Shape: (n_samples,).
        targets: Array of target values. Shape: (n_samples,).
        threshold: Classification threshold. Defaults to 0.5.

    Returns:
        The computed metric as a float in the range [0, 1].

    Raises:
        ValueError: If predictions and targets have mismatched shapes.

    Note:
        This metric is case-sensitive and requires aligned data.

    Example:
        Compute metric on sample data::

            pred = np.array([0.9, 0.1, 0.8])
            true = np.array([1, 0, 1])
            metric = compute_metric(pred, true, threshold=0.5)
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} "
            f"!= targets {targets.shape}"
        )
    # Implementation here
    return result
```

### Docstring Sections

Use these sections as needed, in this order:

1. **Short summary** (1 line): What does this do?
2. **Extended description** (if needed): More detail, context, or behavior.
3. **Args**: Function/method parameters (for functions/methods).
4. **Returns**: Return value description (for functions/methods).
5. **Raises**: Exceptions that may be raised (if applicable).
6. **Attributes**: Class attributes (for classes).
7. **Note** or **Notes**: Additional important information.
8. **Example** or **Examples**: Executable example(s) showing usage.

### Type Hints

Always include type hints in function signatures:

```python
def my_function(x: np.ndarray, y: int = 5) -> Tuple[float, np.ndarray]:
    """Do something with x and y.

    Args:
        x: Input array.
        y: Integer parameter.

    Returns:
        A tuple of (scalar_result, array_result).
    """
    pass
```

### Validating Docstrings

Check docstring compliance with Google-style rules:
```bash
uv run pydocstyle --convention=google footix
```

## Submitting Changes

### Before Submitting

1. Ensure all tests pass:
   ```bash
   uv run pytest -v --cov=footix
   ```

2. Ensure code is formatted and passes linting:
   ```bash
   uv run pre-commit run --all-files
   ```

3. Build and check documentation:
   ```bash
   uv run sphinx-build -b html -W docs/source docs/build/html
   ```

4. Update or add tests for your changes.

5. Update docstrings using Google-style format.

### Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: description of change"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/my-feature
   ```

4. Open a Pull Request on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - A summary of changes

## Code Style Guide

- **Line length**: 99 characters (Ruff default)
- **Python version**: 3.12+
- **Import style**: Organized by Ruff (isort-compatible)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style (as described above)

## CI/CD Pipeline

The CI pipeline runs automatically on pull requests and includes:

- **Linting**: Ruff
- **Type checking**: MyPy
- **Tests**: Pytest with coverage
- **Documentation**: Sphinx build with warnings-as-errors

All checks must pass before merging.

## Questions?

If you have questions, feel free to open an issue or discussion on GitHub.

---

Thank you for contributing to Footix!
