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
import platform
import re
import shutil
from pathlib import Path
from glob import glob

import nbformat as nbf
from _stat import S_IWRITE

from cadetrdm import Study, Case, Options, Environment, ProjectRepo

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
nb_execution_mode = "auto"
nb_execution_excludepatterns = ["example/**/*", "example*"]
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


# -- Functions to load studies-------------------------------------------------

def delete_path(filename):
    def remove_readonly(func, path, exc_info):
        # Clear the readonly bit and reattempt the removal
        # ERROR_ACCESS_DENIED = 5
        from _stat import S_IWRITE
        if func not in (os.unlink, os.rmdir) or exc_info[1].winerror != 5:
            raise exc_info[1]
        os.chmod(path, S_IWRITE)
        func(path)

    absolute_path = os.path.abspath(filename)
    if not os.path.exists(absolute_path):
        return
    if os.path.isdir(absolute_path):
        shutil.rmtree(absolute_path, onerror=remove_readonly)
    else:
        os.remove(absolute_path)


def load_study_results(name, study_url, study_branch):
    project_repo = ProjectRepo(path=study_root / name, url=study_url, branch=study_branch)
    project_repo.update()
    case = Case(project_repo=project_repo, options=options, environment=env)
    # Get the Path to the results of the study.
    # Getting this attribute also loads the results into the cache at the location "path"
    path = case.results_path
    if path is None:
        raise Exception(f"No results found for study {study_url}")

    # Delete previous results
    delete_path(Path("./examples/") / name)
    # Copy new files over. On windows: skip comparison plots as these generate file names that are too long for Windows.
    if platform.system() == 'Windows':
        def ignore(src, names):
            return [_name for _name in names if "_comparison.png" in _name]
    else:
        ignore = None
    shutil.copytree(
        path / "src",
        Path("./examples/") / name,
        ignore=ignore
    )


def set_execution_mode_to_off(notebooks):
    # Set notebook metadata execution_mode to off to ensure these notebooks are not executed again during documentation building
    for ipath in notebooks:
        os.chmod(ipath, S_IWRITE)
        # load in the notebook content with nbformat (nbf)
        ntbk = nbf.read(ipath, nbf.NO_CONVERT)
        # Set execution_mode to off
        ntbk.metadata["mystnb"] = {"execution_mode": "off"}
        # Write modified notebooks out again.
        with open(ipath, "w", encoding="utf-8") as f:
            nbf.write(ntbk, f)


def update_index_with_study_tocs(toc_contents):
    # load in the main index.md to add the study toc entries.
    index_path = "index.md"
    with open(index_path, "r", encoding="utf-8") as f:
        index_lines = f.readlines()
        content = "".join(index_lines)

    case_study_section_regex = ':caption: Case Studies\\n:hidden:\\n[^`]*```\\n'

    toc_contents_joined = '\n'.join(toc_contents)

    case_study_section_updated = f":caption: Case Studies\n:hidden:\n\n{toc_contents_joined}\n```\n"

    content = re.sub(case_study_section_regex, case_study_section_updated, content)

    print(content)

    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(content)


def get_paths_for_table_of_contents(notebooks):
    # Check if there is an index.ipynb. If so: add it to the table of contents. If not, add all ipynbs to the toc
    index_paths = [ipath for ipath in notebooks if "index.ipynb" in ipath]
    if len(index_paths) > 0:
        return index_paths
    else:
        return notebooks


# -- Load in studies -------------------------------------------------

delete_path('./examples')

# The env variable can be used to specify the exact CADET-Process version to be used for the generation of the results
# e.g.: env = Environment(pip_packages={"CADET-Process": version})
env = None

# Studies is a list of tuples (ssh url, branch of study).
studies = [
    ("git@github.com:cadet/RDM-Example-Characterize-Chromatographic-System.git", "tmp/reduce_optimizer_load"),
    # ("git@github.com:cadet/RDM-Example-Multi-State-Steric-Mass-Action.git", "main"),
    ("git@github.com:cadet/RDM-Example-Batch-Elution.git", "main"),
    ("git@github.com:cadet/RDM-Example-Recycling-Techniques.git", "main"),
    # ("git@github.com:cadet/RDM-Example-Load-Wash-Elute.git", "main"),
]

options = Options()
study_root = Path("../_tmp/study_root")
# running list for ipynbs to be added to the table of contents:
toc_contents = []

# Load the studies
for study_url, study_branch in studies:
    print(study_url)
    name = study_url.split("/")[-1].replace(".git", "").replace("RDM-Example-", "")
    load_study_results(name=name, study_url=study_url, study_branch=study_branch)

    # get paths to all ipynb notebooks
    notebooks = glob((Path("./examples/") / name / "**/*.ipynb").as_posix(), recursive=True)
    notebooks = [Path(ipath).as_posix() for ipath in notebooks]

    # Set notebook metadata execution_mode to off to ensure these notebooks are
    # not executed again during documentation building
    set_execution_mode_to_off(notebooks)

    # Add relevant notebook paths to the table of contents list
    toc_contents.extend(get_paths_for_table_of_contents(notebooks))

update_index_with_study_tocs(toc_contents)
