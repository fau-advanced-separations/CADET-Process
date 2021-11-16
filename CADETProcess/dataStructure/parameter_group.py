from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import Structure

class ParametersGroup(Structure):
    """Base class for grouping parameters and exporting them to a dict.

    Attributes
    ----------
    _parameters : List of strings
        List of paramters to be exported.

    See also
    --------
    Parameter
    Descriptor
    ParameterWrapper
    """
    _parameters = []

    def to_dict(self):
        """dict: Dictionary with names and values of the parameters.
        """
        return {
            param: getattr(self, param) for param in self._parameters
                if getattr(self, param) is not None
        }
    
    @property
    def parameters(self):
        """dict: Dictionary with names and values of the parameters.
        """
        return {
            param: getattr(self, param) for param in self._parameters
                if getattr(self, param) is not None
        }
    
    @parameters.setter
    def parameters(self, parameters):
        for param, value in parameters.items():
            if param not in self._parameters:
                raise CADETProcessError('Not a valid parameter')
            if value is not None:
                setattr(self, param, value)
    

class ParameterWrapper(ParametersGroup):
    """Base class for converting the config from objects such as units.

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

    See also
    --------
    Parameter
    Descriptor
    ParametersGroup
    """
    _base_class = object

    def __init__(self, wrapped_object):
        if not isinstance(wrapped_object, self._baseClass):
            raise CADETProcessError("Expected {}".format(self._baseClass))

        model = wrapped_object.model
        try:
            self.model_parameters = self._model_parameters[model]
        except KeyError:
            raise CADETProcessError("Model Type not defined")

        self._wrapped_object = wrapped_object

    def to_dict(self):
        """Returns the parameters for the model and solver in a dictionary.

        Defines the parameters for the model and solver and saves them into the
        respective dictionary. The cadet_parameters are get by the
        parameters_dict of the inherited functionality of the ParametersGroup.
        The keys for the model_solver_parameters are get by the attributes of
        the value of the wrapped_object, if they are not None. Both, the
        solver_parameters and the model_solver_parameters are saved into the
        parameters_dict.

        Returns
        -------
        parameters_dict : dict
            Dictionary, containing the attributes of each parameter from the
            model_parameters and the cadet_parameters.

        See also
        --------
        ParametersGroup
        """
        solver_parameters = super().to_dict()
        model_parameters = {}

        model_parameters[self._model_type] = self.model_parameters['name']

        for key, value in self.model_parameters['parameters'].items():
            value = getattr(self._wrapped_object, value)
            if value is not None:
                model_parameters[key] = value

        for key, value in self.model_parameters.get('fixed', dict()).items():
            if isinstance(value, list):
                value = self._wrapped_object.n_comp * value
            model_parameters[key] = value

        return {**solver_parameters, **model_parameters}
    