"""Test suite for documentation build integrity.

This module verifies that Sphinx builds successfully and treats warnings as errors.

"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_sphinx_build() -> None:
    """Test that Sphinx builds docs without warnings or errors.

    Runs `sphinx-build -b html -W docs/source docs/build/html` and fails if any
    warnings or errors occur. This ensures that the API documentation and
    docstring formatting remain valid.

    Raises:
        AssertionError: If Sphinx build fails or warnings are treated as errors.

    """
    repo_root = Path(__file__).parent.parent
    docs_source = repo_root / "docs" / "source"
    docs_build = repo_root / "docs" / "build" / "html"
    docs_build_root = repo_root / "docs" / "build"

    # Clean the build cache to ensure a fresh build (avoids state-dependent test failures)
    import shutil

    if docs_build_root.exists():
        shutil.rmtree(docs_build_root)

    # Treat every Sphinx warning as a test failure.
    build_cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
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
            f"Sphinx build failed with exit code {result_build.returncode}.\n"
            f"Check output above for errors."
        )
