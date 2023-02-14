"""
===================================================
Data Structures (:mod:`CADETProcess.dataStructure`)
===================================================

.. currentmodule:: CADETProcess.dataStructure

This module provides datastructures to simplify defining setters and getters for
(model) parameters.
It is mostly based on the data model introduced in [1]_

Note
----

At some point it might be considered to switch to attrs
(see `#15 <https://github.com/fau-advanced-separations/CADET-Process/issues/15>`_).

References
----------
.. [1] Jones, B. K., Beazley, D. (2013).
    Python Cookbook: Recipes for Mastering Python 3. United States: O'Reilly Media.


.. autosummary::
    :toctree: generated/

    dataStructure
    parameter
    parameter_group
    cache
    diskcache
    nested_dict

"""

from .dataStructure import *
from .parameter import *
from .parameter_group import *
from .cache import *
from .diskcache import *
from .nested_dict import *
