# v0.7.0

**CADET-Process** v0.7.0 is the culmination of 10 months of hard work.
It contains many new features, numerous bug-fixes, improved test coverage and better
documentation.
There have been a number of deprecations and API changes in this release, which are documented below.
All users are encouraged to upgrade to this release, as there are a large number of bug-fixes and
optimizations.
Before upgrading, we recommend that users check that their own code does not use deprecated CADET-Process functionality (to do so, run your code with ``python -Wd`` and check for ``DeprecationWarning`` s).

This release requires Python 3.8+


## Highlights of this release


## New features

### {mod}`CADETProcess.processModel` improvements


### {mod}`CADETProcess.comparison` improvements


### {mod}`CADETProcess.fractionation` improvements


### {mod}`CADETProcess.optimization` improvements


## Deprecated features
The following functions will be removed in a future release.
Users are suggested to upgrade to [...]


## Expired Deprecations
The following previously deprecated features are now removed:
- Change 1
- Change 2

## Other changes
- Change 1
- Change 2


## Issues closed for 0.7.0
- [2](https://github.com/fau-advanced-separations/CADET-Process/issues/2): Add Parameter Sensitivities
- [10](https://github.com/fau-advanced-separations/CADET-Process/issues/10): Update Pymoo Interface to v0.6.0


## Pull requests for 0.7.0
- [13](https://github.com/fau-advanced-separations/CADET-Process/pull/13): Change normalization for NRMSE from max solution to max reference
- [16](https://github.com/fau-advanced-separations/CADET-Process/pull/16): Overhaul docs
