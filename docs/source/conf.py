# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "footix"
copyright = "2025, Shaheen Acheche"
author = "Shaheen Acheche"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # auto-generate API docs
    "sphinx.ext.napoleon",  # Google & NumPy style docstrings
    "sphinx.ext.viewcode",  # link to highlighted source
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",  # link to external docs (numpy, pandas, python)
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo_footix.png"
html_theme_options = {
    "logo_only": False,
    # set to True to collapse navigation on initial load
    "collapse_navigation": False,
    # scroll depth for the left sidebar tree
    "navigation_depth": 1,
    "style_nav_header_background": "#0C192A",
}

# Napoleon (Google-style docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# Autodoc and typehints
autodoc_member_order = "bysource"  # or 'alphabetical'
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}

# Intersphinx: link references to external docs (Python, NumPy, Pandas)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Enforce strict docstring and cross-reference checks
nitpicky = True
