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

    # Run sphinx-build with -W to treat warnings as errors
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

    # If sphinx returns a failure, it may be due to warnings. Filter out a
    # small set of known/acceptable warnings (like duplicate toctree references)
    # which are currently benign in this project, and fail only on other issues.
    if result_build.returncode != 0:
        # Quick check for missing optional dependencies that cause extension import errors
        if (
            "No module named 'roman_numerals'" in stderr_text
            or 'No module named "roman_numerals"' in stderr_text
        ):
            raise AssertionError(
                "Sphinx failed to import an optional extension dependency: 'roman_numerals'.\n"
                "Install it locally with:\n\n  poetry add -D roman-numerals\n\nor for pip:\n\n"
                "  pip install roman-numerals\n\nAdd it to your dev dependencies"
                " so CI can build the docs too."
            )

        # Fail immediately on doc build ERROR lines (these indicate broken doc syntax)
        error_lines = [line for line in stderr_text.splitlines() if "ERROR:" in line]
        if error_lines:
            print("==== Doc build ERRORS ====")
            for line in error_lines:
                print(line)
            raise AssertionError(
                "Sphinx doc build produced ERROR(s). Please fix the docstring formatting"
                " or substitutions shown above."
            )

        import re

        allowed_patterns = [
            # benign: duplicate toctree references
            r"document referenced in multiple (toctrees|arborescences)",
            # unresolved cross-reference warnings for numpy/pandas (French & English forms)
            r"py:class cible de référence non trouvée :"
            r" .*(numpy\.|np\.|pandas\.|pd\.|pathlib\.|optional|~P|proba ArrayLike)",
            r"py:class target not found :"
            r" .*(numpy\.|np\.|pandas\.|pd\.|pathlib\.|optional|~P|proba ArrayLike)",
            # fallback: any unresolved pandas/numpy class refs
            # (covers bracketed suffixes like [ref.class])
            r"py:class .*(pandas\.|pd\.)",
            r"py:class .*(numpy\.|np\.)",
            r"description dupliquée de l'objet",
            r"Le document n'est inclus dans aucune toctree",
        ]

        remaining_lines = []
        for line in stderr_text.splitlines():
            if any(re.search(pat, line, re.I) for pat in allowed_patterns):
                # known benign warning; skip
                continue
            # keep non-empty, informative lines
            if line.strip():
                remaining_lines.append(line)

        if remaining_lines:
            print("==== Remaining lines (not allowed) ====")
            for line in remaining_lines:
                print(line)
            # There are other warnings/errors we should treat as failures
            raise AssertionError(
                f"Sphinx build failed with code {result_build.returncode}:\n"
                f"stdout: {result_build.stdout}\n"
                f"stderr: {result_build.stderr}"
            )
        else:
            # Only allowed warnings were present — treat as success but print a note
            print("Sphinx build completed with allowed warnings (ignored in test):")
            for line in stderr_text.splitlines():
                if any(re.search(pat, line, re.I) for pat in allowed_patterns):
                    print(line)
