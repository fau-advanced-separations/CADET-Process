# v0.7.0


**CADET-Process** v0.7.0 is the culmination of 10 months of hard work.
It contains many new features, numerous bug-fixes, improved test coverage and better
documentation.
There have been a number of deprecations and API changes in this release, which are documented below.
All users are encouraged to upgrade to this release, as there are a large number of bug-fixes and
optimizations.
Before upgrading, we recommend that users check that their own code does not use deprecated CADET-Process functionality (to do so, run your code with ``python -Wd`` and check for ``DeprecationWarning`` s).

This release requires Python 3.8+ and NumPy 1.19.5 or greater.


## Highlights of this release
- A new dedicated datasets submodule (`scipy.datasets`) has been added, and is
  now preferred over usage of `scipy.misc` for dataset retrieval.
- A new `scipy.interpolate.make_smoothing_spline` function was added. This
  function constructs a smoothing cubic spline from noisy data, using the
  generalized cross-validation (GCV) criterion to find the tradeoff between
  smoothness and proximity to data points.
- `scipy.stats` has three new distributions, two new hypothesis tests, three
  new sample statistics, a class for greater control over calculations
  involving covariance matrices, and many other enhancements.


## New features

### {mod}`CADETProcess.processModel` improvements

### {mod}`CADETProcess.comparison` improvements

### {mod}`CADETProcess.fractionation` improvements
- Improved random variate sampling of several distributions.

  - Drawing multiple samples from `scipy.stats.matrix_normal`,
    `scipy.stats.ortho_group`, `scipy.stats.special_ortho_group`, and
    `scipy.stats.unitary_group` is faster.
  - The ``rvs`` method of `scipy.stats.vonmises` now wraps to the interval
    ``[-np.pi, np.pi]``.
  - Improved the reliability of `scipy.stats.loggamma` ``rvs`` method for small
    values of the shape parameter.

### {mod}`CADETProcess.optimization` improvements


## Deprecated features
- `scipy.misc` module and all the methods in ``misc`` are deprecated in v1.10
  and will be completely removed in SciPy v2.0.0. Users are suggested to
  utilize the `scipy.datasets` module instead for the dataset methods.


## Expired Deprecations
- There is an ongoing effort to follow through on long-standing deprecations.
- The following previously deprecated features are affected:
  - Removed ``cond`` & ``rcond`` kwargs in ``linalg.pinv``
  - Removed wrappers ``scipy.linalg.blas.{clapack, flapack}``


## Other changes
- Change 1
- Change 2


## Issues closed for 0.7.0
- [1261](https://github.com/scipy/scipy/issues/1261): errors in fmin_bfgs and some improvements


## Pull requests for 0.7.0
- [1261](https://github.com/scipy/scipy/pulls/1261): errors in fmin_bfgs and some improvements
