# Contributing to **CADET-Process**

Thank you for your interest in contributing to **CADET-Process**!
To streamline the development process, please carefully read the following guide.
If you need further guidance, you can find our team on the [CADET-Forum](https://forum.cadet-web.de/).

## Environment Setup and Dependency Installation

To contribute to **CADET-Process**, we recommend using [conda-forge](https://conda-forge.org/) to manage your environment.
This is necessary because **CADET-Core** is distributed via **conda-forge**.
Otherwise, you will need to [build and install it from source](https://cadet.github.io/master/getting_started/installation_core.html#install-from-source).

A recommended conda environment file would look like this:

```yaml
name: cadet-process
channels:
  - conda-forge
dependencies:
  - python >=3.12,<3.13
  - cadet >=5.0.3
  - pip
```

Use the terminal to set up the environment:

```bash
conda env create -f cadet-process.yml
conda activate cadet-process
```

Clone the **CADET-Process** repository to your local machine and navigate to the directory:

```bash
git clone https://github.com/fau-advanced-separations/CADET-Process.git
cd CADET-Process
```

Install **CADET-Process** as an editable package within your `conda` environment via `pip`:

```bash
pip install -e . --group dev
```

Note: The `dev` group installs additional dependencies required for linting, formatting, testing, and building the documentation (as described below).


### Updating Python Versions

The shared Conda environment allows Python 3.12 through 3.14 (`>=3.12,<3.15`).
Read the Docs uses this range and may select its newest available version, so keep an explicit upper bound to prevent an untested Python release from breaking dependency installation.
The package's `requires-python` setting is a separate compatibility declaration; it currently permits Python 3.10 and newer, while CI tests 3.12 through 3.14.

When changing the Python versions, review these settings together:

| File | Setting and purpose |
| --- | --- |
| `environment.yml` | Python range for the shared Conda environment, including Read the Docs and test CI. |
| `.readthedocs.yaml` | Selects Miniforge, loads `environment.yml`, and installs the `all` extra and `docs` dependency group. The Python interpreter range comes from `environment.yml`. |
| `pyproject.toml` | `project.requires-python` declares package compatibility; dependency constraints and optional extras must work on the intended versions. |
| `.github/workflows/pipeline.yml` and `.github/workflows/scheduled-tests.yml` | Python test matrices, including the Windows and macOS entries. Each job pins its matrix version before loading `environment.yml`. |
| `.github/workflows/docs.yml` | Explicit Python version for the GitHub documentation build. |
| `.github/workflows/coverage.yml` | Python version used for coverage. |
| `.github/workflows/publish-to-pypi.yml` | Interpreter used to build release distributions; currently follows the latest Python 3 via `3.x`. |
| `CONTRIBUTING.md` | Recommended development environment above and this version-maintenance note. The example may use one version within the shared range. |

Before raising the upper bound, verify that required dependencies, optional extras, and CADET-Core are available for the new Python version on the relevant platforms.
Check binary wheel availability for compiled pip dependencies, since a missing wheel can trigger a source build requiring additional compilers and libraries.
Run the test matrices and both documentation builds before treating the new version as validated.
The GitHub documentation job disables notebook execution, while Read the Docs uses cached execution, so a passing GitHub build alone does not validate the Read the Docs environment.
Change `requires-python` only when the package's compatibility policy changes, rather than automatically copying the documentation environment range into it.

## Code Requirements

The **CADET-Process** codebase maintains a high standard for code quality, which we enforce via GitHub Actions.

### Coding Style

We use the [`Ruff` linter / formatter](https://docs.astral.sh/ruff/formatter/) for all Python files.
To install the latest version of Ruff, use the following command:


```bash
pip install ruff
```

**Linting:** To check your files for compliance with our coding standards, run the following command:

```bash
ruff check CADETProcess
```

This command will analyze the specified directory and report any issues that need to be addressed.

**Formatting:** To automatically format your files according to our coding standards, use:

```bash
ruff format CADETProcess
```

This command will apply the necessary formatting changes to ensure consistency across the codebase.

### Unit Tests

The majority of our code is covered by unit tests, and we are working to achieve 100% code coverage.
Please ensure that new code is covered by unit tests.

To run all unit tests, install the test dependencies via:

```bash
pip install -e . --group testing
```

Then run:

```bash
pytest tests --rootdir=tests
```

Note: Some tests can take a long time to run. To exclude slow tests, run:

```bash
pytest tests --rootdir=tests -m "not slow"
```

To analyze test coverage, run:

```bash
pytest --cov=./CADETProcess
```

### Pre-commit Hooks

Contributors can use [pre-commit](https://pre-commit.com/) to run `ruff` and other hooks as part of the commit process.
To install the hooks, first install `pre-commit` via:

```bash
pip install pre-commit
```

Then run:

```bash
pre-commit install
```

from the repository root.

## Issues vs. Pull Requests

We use **GitHub issues** for:

* Reporting bugs
* Suggesting new features
* Discussing technical problems and ideas

Please use an appropriate provided template and ensure your description and instructions are clear.

**Pull Requests** should be submitted after your changes are complete and tested.
Make sure your PR addresses one issue or feature to keep reviews manageable.
Reference related issues in your PR description.
Add an entry to the release notes of the upcoming version (`docs/source/release_notes/<version>.md`) for every user-visible change: new features, bug fixes, behavior changes, and deprecations.
Describe the change from the user's perspective and link the pull request or issue.

## Branch Policy

We use the [conventional branch](https://conventional-branch.github.io/) name convention to keep contributions organized:

- `master`: The release branch, updated only when a version is tagged
- `dev`: The main development branch
- `feature/`: For new features (e.g., `feature/add-new-binding-model`)
- `fix/`: For bug fixes (e.g., `fix/fix-header-bug`, `fix/header-bug`)
- `hotfix/`: For urgent fixes (e.g., `hotfix/security-patch)`
- `release/`: For branches preparing a release (e.g., `release/v1.2.0)`
- `chore/`: For non-code tasks like dependency, docs updates (e.g., `chore/update-dependencies)`

## Documentation

We require docstrings for all public functions and classes (i.e., those not starting with an underscore `_`).
We use the [Numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings and [Sphinx](https://www.sphinx-doc.org/) to compile the documentation.

To build the documentation locally, install the necessary dependencies:

```bash
pip install -e . --group docs
```

Then, in the `docs` directory, run:

```bash
sphinx-build -b html source build
```

The output will be located in the `build` directory and can be opened with any browser.

## Deprecating API

Use the `@deprecated(deprecated_in, removed_in, use)` decorator from `CADETProcess.dataStructure.deprecation` to deprecate a function, method, or property.
It emits a versioned `DeprecationWarning`, sets the PEP 702 type-checker flag, and appends a rendered `.. deprecated::` directive to the docstring, so the deprecation shows up in the API reference automatically without hand-authoring it.
Every decorated callable is registered internally, and `tests/dataStructure/test_deprecation.py` fails once its `removed_in` version has shipped, so an overdue removal can't go unnoticed.
To deprecate a single keyword argument rather than the whole callable, use `@deprecated_alias(deprecated_in, removed_in, old_name="new_name")` instead.

## Publish Package

**CADET-Process** is automatically published to [PyPI](https://pypi.org/project/CADET-Process) when a GitHub release is created.
Make sure to include relevant information in the changelog.
