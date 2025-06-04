# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'footix'
copyright = '2025, Shaheen Acheche'
author = 'Shaheen Acheche'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # auto-generate API docs
    'sphinx.ext.napoleon',          # Google & NumPy style docstrings
    'sphinx.ext.viewcode',          # link to highlighted source
#    'sphinx_autodoc_typehints',     # show Python type hints
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    # set to True to collapse navigation on initial load
    'collapse_navigation': False,
    # scroll depth for the left sidebar tree
    'navigation_depth': 4,  
}
napoleon_google_docstring = True

autodoc_member_order = 'bysource'  # or 'alphabetical'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}
