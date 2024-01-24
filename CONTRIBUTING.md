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

## Branch Policy

We use a branch naming convention to keep contributions organized:

* **Feature branches**: `feature/your-feature-name`
* **Bug fixes**: `fix/short-description`
* **Hotfixes**: `hotfix/issue-description`
* **Documentation updates**: `docs/update-description`

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

## Publish Package

**CADET-Process** is automatically published to [PyPI](https://pypi.org/project/CADET-Process) when a GitHub release is created.
Make sure to include relevant information in the changelog.
