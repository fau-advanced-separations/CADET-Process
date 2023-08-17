import numpy as np

from .dataStructure import Aggregator


class SizedAggregator(Aggregator):
    """Aggregator for sized parameters."""

    def _parameter_shape(self, instance):
        values = self._get_parameter_values_from_container(instance)
        shapes = [np.array(el, ndmin=1).shape for el in values]

        if len(set(shapes)) > 1:
            raise ValueError("Inconsistent parameter shapes.")

        if len(shapes) == 0:
            return ()

        return shapes[0]

    def _expected_shape(self, instance):
        return (self._n_instances(instance), ) + self._parameter_shape(instance)

    def _get_parameter_values_from_container(self, instance):
        value = super()._get_parameter_values_from_container(instance)

        if value is None or len(value) == 0:
            return

        value = np.array(value, ndmin=2).T
        return value

    def _check(self, instance, value, recursive=False):
        expected_shape = self._expected_shape(instance)
        if value.shape != expected_shape:
            raise ValueError(
                f"Expected a array with shape {expected_shape}, got {value.shape}"
            )

        if recursive:
            super()._check(instance, value, recursive)

    def _prepare(self, instance, value, recursive=False):
        value = np.array(value)

        if recursive:
            value = super()._prepare(instance, value, recursive)

        return value


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

    def _get_parameter_values_from_container(self, instance):
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

        return values

class SizedClassDependentAggregator(SizedAggregator, ClassDependentAggregator):
    pass
