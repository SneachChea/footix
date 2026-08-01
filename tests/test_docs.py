"""Test suite for documentation build integrity.

This module verifies that Sphinx builds successfully and treats warnings as errors.

"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _build(builder: str, docs_source: Path, docs_build: Path) -> None:
    """Run a strict Sphinx build of a given builder, failing on any warning."""
    if docs_build.exists():
        shutil.rmtree(docs_build)

    build_cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        builder,
        "-W",
        str(docs_source),
        str(docs_build),
    ]
    result_build = subprocess.run(build_cmd, capture_output=True, text=True)

    # Always print Sphinx build output for debugging
    stderr_text = (result_build.stderr or "") + "\n" + (result_build.stdout or "")
    print("==== Sphinx build output (stdout + stderr) ====")
    print(stderr_text)

    # Fail if Sphinx reports an error or warning.
    if result_build.returncode != 0:
        raise AssertionError(
            f"Sphinx {builder} build failed with exit code {result_build.returncode}.\n"
            f"Check output above for errors."
        )


@pytest.mark.slow
def test_sphinx_build() -> None:
    """Test that Sphinx builds docs without warnings or errors.

    Runs `sphinx-build -b html -W docs/source docs/build/html` and fails if any
    warnings or errors occur. This ensures that the API documentation and
    docstring formatting remain valid. Also checks that the tutorial notebooks
    are exposed as downloadable files.

    Raises:
        AssertionError: If Sphinx build fails or warnings are treated as errors.

    """
    repo_root = Path(__file__).parent.parent
    docs_source = repo_root / "docs" / "source"
    docs_build = repo_root / "docs" / "build" / "html"
    docs_build_root = repo_root / "docs" / "build"

    # Clean the build cache to ensure a fresh build (avoids state-dependent test failures)
    if docs_build_root.exists():
        shutil.rmtree(docs_build_root)

    _build("html", docs_source, docs_build)

    downloads_dir = docs_build / "_downloads"
    downloaded_notebooks = {path.name for path in downloads_dir.rglob("*.ipynb")}
    assert {"elo.ipynb", "poisson.ipynb"}.issubset(
        downloaded_notebooks
    ), f"Expected tutorial notebooks in {downloads_dir}, got {downloaded_notebooks}"


@pytest.mark.slow
def test_sphinx_doctest_build() -> None:
    """Test that all doctest examples in the docs run successfully.

    Runs `sphinx-build -b doctest -W docs/source docs/build/doctest` and fails
    if any example fails or any warning occurs.

    Raises:
        AssertionError: If the doctest build fails.

    """
    repo_root = Path(__file__).parent.parent
    docs_source = repo_root / "docs" / "source"
    docs_build = repo_root / "docs" / "build" / "doctest"

    _build("doctest", docs_source, docs_build)
