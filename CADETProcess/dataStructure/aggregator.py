import numpy as np

from .dataStructure import Aggregator

import numpy as np


class NumpyProxyArray(np.ndarray):
    """A numpy array that dynamically updates attributes of container elements."""

    def __new__(cls, aggregator, instance):
        values = aggregator._get_values_from_container(instance, transpose=True)

        if values is None:
            return

        obj = values.view(cls)

        obj.aggregator = aggregator
        obj.instance = instance

        return obj

    def _get_values_from_aggregator(self):
        """Refresh data from the underlying container."""
        return self.aggregator._get_values_from_container(
            self.instance, transpose=True, check=True
        )

    def __getitem__(self, index):
        """Retrieve an item from the aggregated parameter array."""
        return self._get_values_from_aggregator()[index]

    def __setitem__(self, index, value):
        """
        Modify an individual element in the aggregated parameter list.
        This ensures changes are propagated back to the objects.
        """
        current_value = self._get_values_from_aggregator()
        current_value[index] = value
        self.aggregator.__set__(self.instance, current_value)

    def __array_finalize__(self, obj):
        """Ensure attributes are copied when creating a new view or slice."""
        if obj is None:
            self.aggregator = None
            self.instance = None
            return

        if not isinstance(obj, NumpyProxyArray):
            return np.asarray(obj)

        self.aggregator = getattr(obj, 'aggregator', None)
        self.instance = getattr(obj, 'instance', None)

    def __array_function__(self, func, types, *args, **kwargs):
        """
        Ensures that high-level NumPy functions (like np.dot, np.linalg.norm) return a
        normal np.ndarray.
        """
        result = super().__array_function__(func, types, *args, **kwargs)
        return np.asarray(result)

    def __repr__(self):
        """Return a fresh representation that reflects live data."""
        return f"NumpyProxyArray({self._get_values_from_aggregator().__repr__()})"


class SizedAggregator(Aggregator):
    """Aggregator for sized parameters."""

    def __init__(self, *args, transpose=False, **kwargs):
        """
        Initialize a SizedAggregator instance.

        Parameters
        ----------
        *args : Any
            Variable length argument list.
        transpose : bool, options
            If False, the parameter shape will be ((n_instances, ) + parameter_shape).
            Else, it will be (parameter_shape + (n_instances, ))
            The default is False.
        **kwargs : Any
            Arbitrary keyword arguments.
        """
        self.transpose = transpose

        super().__init__(*args, **kwargs)

    def _parameter_shape(self, instance):
        values = self._get_values_from_container(instance, transpose=False)

        shapes = [el.shape for el in values]

        if len(set(shapes)) > 1:
            raise ValueError("Inconsistent parameter shapes.")

        if len(shapes) == 0:
            return ()

        return shapes[0]

    def _expected_shape(self, instance):
        if self.transpose:
            return self._parameter_shape(instance) + (self._n_instances(instance), )
        else:
            return (self._n_instances(instance), ) + self._parameter_shape(instance)

    def _get_values_from_container(self, instance, transpose=False, check=False):
        value = super()._get_values_from_container(instance, check=False)

        if value is None or len(value) == 0:
            return

        value = np.array(value, ndmin=2)

        if check:
            value = self._prepare(instance, value, transpose=False, recursive=True)
            self._check(instance, value, transpose=True, recursive=True)

        if transpose and self.transpose:
            value = value.T

        return value

    def _check(self, instance, value, transpose=True, recursive=False):
        value_array = np.array(value, ndmin=2)
        if transpose and self.transpose:
            value_array = value_array.T

        value_shape = value_array.shape
        expected_shape = self._expected_shape(instance)

        if value_shape != expected_shape:
            raise ValueError(
                f"Expected a array with shape {expected_shape}, got {value_shape}"
            )

        if recursive:
            super()._check(instance, value, recursive)

    def _prepare(self, instance, value, transpose=False, recursive=False):
        value = np.array(value, ndmin=2)

        if transpose and self.transpose:
            value = value.T

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value

    def __get__(self, instance, cls):
        """
        Retrieve the descriptor value for the given instance.

        Parameters
        ----------
        instance : Any
            Instance to retrieve the descriptor value for.
        cls : Type[Any], optional
            Class to which the descriptor belongs.

        Returns
        -------
        NumpyProxyArray
            A numpy-like array that modifies the underlying objects when changed.
        """
        if instance is None:
            return self

        return NumpyProxyArray(self, instance)

    def __set__(self, instance, value):
        """
        Set the descriptor value for the given instance.

        Parameters
        ----------
        instance : Any
            Instance to set the descriptor value for.
        value : Any
            Value to set.
        """
        value = self._prepare(instance, value, transpose=True)
        super().__set__(instance, value)


class ClassDependentAggregator(Aggregator):
    """Aggregator where parameter name changes depending on instance type."""

    def __init__(self, *args, mapping, **kwargs):
        """
        Initialize the Aggregator descriptor.

        Parameters
        ----------
        mapping : dict
            Mapping of instance types and parameter names.
        *args : tuple, optional
            Additional positional arguments.
        **kwargs : dict, optional
            Additional keyword arguments.

        """
        self.mapping = mapping

        super().__init__(*args, **kwargs)

    def _get_values_from_container(self, instance, check=False):
        container = self._container_obj(instance)

        values = []
        for el in container:
            if type(el) in self.mapping:
                attr = self.mapping[type(el)]
            else:
                attr = self.mapping[None]
                if attr is None:
                    continue

            value = getattr(el, attr)
            values.append(value)

        if len(values) == 0:
            return

        if check:
            values = self._prepare(instance, values, transpose=False, recursive=True)
            self._check(instance, values, transpose=True, recursive=True)

        return values


class SizedClassDependentAggregator(SizedAggregator, ClassDependentAggregator):
    pass
