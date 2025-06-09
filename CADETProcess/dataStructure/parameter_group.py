from typing import Any

from CADETProcess import CADETProcessError


class ParameterWrapper:
    """
    Base class for converting the config from objects such as units.

    Attributes
    ----------
    _base_class : type
        Type constraint for wrapped object
    _wrapped_object : obj
        Object whose config is to be converted

    Raises
    ------
    CADETProcessError
        If the wrapped_object is no instance of the base_class.
    """

    _base_class = object

    def __init__(self, wrapped_object: Any) -> None:
        """Construct ParameterWrapper object."""
        if not isinstance(wrapped_object, self._baseClass):
            raise CADETProcessError(f"Expected {self._baseClass}")

        model = wrapped_object.model
        try:
            self.model_parameters = self._model_parameters[model]
        except KeyError:
            raise CADETProcessError("Model Type not defined")

        self._wrapped_object = wrapped_object

    @property
    def parameters(self) -> dict:
        """dict: Parameters dictionary."""
        model_parameters = {}

        model_parameters[self._model_type] = self.model_parameters["name"]

        for key, value in self.model_parameters["parameters"].items():
            value = getattr(self._wrapped_object, value)
            if value is not None:
                model_parameters[key] = value

        for key, value in self.model_parameters.get("fixed", dict()).items():
            if isinstance(value, list):
                value = self._wrapped_object.n_comp * value
            model_parameters[key] = value

        return model_parameters
