import json
from typing import Any

import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.

    This encoder extends `json.JSONEncoder` to support serialization of NumPy integers,
    floating-point numbers, and arrays. It converts NumPy data types to native Python
    data types that can be serialized to JSON.
    """

    def default(self, obj: Any) -> Any:
        """
        Convert NumPy data types to native Python data types for JSON serialization.

        Parameters
        ----------
        obj : Any
            The object to serialize.

        Returns
        -------
        Any
            The object converted to a JSON-serializable type.

        Raises
        ------
        TypeError
            If the object is not of a supported type.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)
