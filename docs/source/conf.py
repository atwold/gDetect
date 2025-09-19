# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Add project path to sys.path --------------------------------------------
import os
import sys
project_path = os.path.abspath('../../src')
if project_path not in sys.path: sys.path.insert(0, project_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gDetect'
copyright = '2025, Alexander Wold'
author = 'Alexander Wold'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # Supports NumPy & Google-style docstrings
    "sphinx_autodoc_typehints",     # Parses type hints
    "numpydoc",
    "sphinx.ext.viewcode",
    'sphinx.ext.intersphinx'
]

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
