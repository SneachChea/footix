# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath("../"))

project = "footix"
copyright = "2025, Shaheen Acheche"
author = "Shaheen Acheche"
try:
    release = package_version("pyfootix")
except PackageNotFoundError:
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # auto-generate API docs
    "sphinx.ext.napoleon",  # Google & NumPy style docstrings
    "sphinx.ext.viewcode",  # link to highlighted source
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
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# Autodoc and typehints
autodoc_member_order = "bysource"  # or 'alphabetical'
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

# Intersphinx: link references to external docs (Python, NumPy, Pandas)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

nitpick_ignore = [
    ("py:class", "Path"),
    ("py:class", "np.array"),
    ("py:class", "np.ndarray"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "matplotlib.figure.Figure"),
    ("py:class", "xarray.core.datatree.DataTree"),
    ("py:class", "arviz.InferenceData"),
    ("py:class", "footix.data_io.prediction_export.PredictionExportModel"),
    ("py:class", "ArrayLike"),
    ("py:class", "arraylike"),
    ("py:class", "shape"),
    ("py:class", "3"),
    ("py:class", "default=1e-12"),
    ("py:class", "default=50"),
    ("py:class", "proba ArrayLike"),
    ("py:class", "optional"),
    ("py:class", "Defaults to"),
    ("py:class", "on a single bet. Defaults to"),
    ("py:class", "all bets. Defaults to"),
    ("py:class", "None. The method updates the instance attributes"),
    ("py:class", "~P"),
    ("py:class", "P"),
    ("py:class", "R"),
    ("py:class", "footix.utils.decorators.R"),
    ("py:class", "selected. Defaults to"),
    ("py:exc", "pandas.read_csv"),
    ("py:exc", "filesystem operations."),
]

# Enforce strict docstring and cross-reference checks
nitpicky = True
