# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from datetime import date

# -- Project information -----------------------------------------------------

project = 'CADET-Process'
copyright = f'2019-{date.today().year}'
author = 'Johannes Schmölder'

import CADETProcess
version = CADETProcess.__version__
release = CADETProcess.__version__.replace("_", "")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Extensions
extensions = []

## MyST-NB
extensions.append("myst_nb")
nb_execution_mode = "cache"
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'myst-nb',
}

## Numpydoc
extensions.append("numpydoc")
numpydoc_class_members_toctree = False

## Autodoc
extensions.append("sphinx.ext.autodoc")

## Autosummary
extensions.append("sphinx.ext.autosummary")
autosummary_generate = True

## Intersphinx mapping
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "cadet": ("https://cadet.github.io/master/", None),
}

## To do
extensions.append("sphinx.ext.todo")
todo_include_todos = True

## Viewcode
extensions.append("sphinx.ext.viewcode")

## Copy Button
extensions.append("sphinx_copybutton")

## BibTeX
extensions.append("sphinxcontrib.bibtex")
bibtex_bibfiles = ['references.bib']

# -- Internationalization ------------------------------------------------
# specifying the natural language populates some key tags
language = "en"

# ReadTheDocs has its own way of generating sitemaps, etc.
if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]

    # -- Sitemap -------------------------------------------------------------
    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Extension options -------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"

html_theme_options = {
    "show_toc_level": 2,
    "use_download_button": True,
    "repository_url": "https://github.com/fau-advanced-separations/CADET-Process",
	"use_repository_button": True,
    "use_issues_button": True,
}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Copy examples
import shutil
shutil.rmtree('./examples', ignore_errors=True)
shutil.copytree('../../examples', './examples/')
