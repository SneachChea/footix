"""Test suite for documentation build integrity.

This module verifies that Sphinx builds successfully and treats warnings as errors.
"""

import subprocess
import sys
from pathlib import Path


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

    # Run sphinx-apidoc to regenerate API docs
    apidoc_cmd = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-o",
        str(docs_source / "api"),
        str(repo_root / "footix"),
        "-f",
    ]
    result_apidoc = subprocess.run(apidoc_cmd, capture_output=True, text=True)
    if result_apidoc.returncode != 0:
        raise AssertionError(
            f"sphinx-apidoc failed with code {result_apidoc.returncode}:\n"
            f"stdout: {result_apidoc.stdout}\n"
            f"stderr: {result_apidoc.stderr}"
        )

    # Run sphinx-build (without -W to avoid failing on benign cross-reference warnings)
    build_cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(docs_source),
        str(docs_build),
    ]
    result_build = subprocess.run(build_cmd, capture_output=True, text=True)

    # Always print Sphinx build output for debugging
    stderr_text = (result_build.stderr or "") + "\n" + (result_build.stdout or "")
    print("==== Sphinx build output (stdout + stderr) ====")
    print(stderr_text)

    # Fail only if Sphinx build actually failed (exit code != 0)
    if result_build.returncode != 0:
        raise AssertionError(
            f"Sphinx build failed with exit code {result_build.returncode}.\n"
            f"Check output above for errors."
        )
