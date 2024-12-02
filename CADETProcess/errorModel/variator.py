from typing import Dict

from CADETProcess.errorModel.distribution import DistributionBase


class Variator:
    """
    A class to manage variations in a model using error distributions.

    Attributes
    ----------
    model : Any
        The model object to which variations will be applied.
    error_distributions : dict
        A dictionary to store registered error distributions indexed by name.
    """

    def __init__(self, model):
        """
        Initialize the Variator with a model.

        Parameters
        ----------
        model : Any
            The model object to which variations will be applied.
        """
        self.model = model
        self.error_distributions: Dict[str, VariatedVariable] = {}

    def add_error(self, name: str, parameter_path: str, distribution: DistributionBase):
        """
        Register an error distribution for a specific parameter in the model.

        Parameters
        ----------
        name : str
            The unique name for the error being registered.
        parameter_path : str
            The path to the parameter in the model that the error affects.
            Example: 'layer1.weights[0][0]'
        distribution : DistributionBase
            The distribution object defining the error.

        Raises
        ------
        ValueError
            If the name is already registered.
        """
        if name in self.error_distributions:
            raise ValueError(f"Error with name '{name}' is already registered.")

        self.error_distributions[name] = VariatedVariable(parameter_path, distribution)


class VariatedVariable:
    """
    Encapsulates a parameter path and its associated error distribution.

    Attributes
    ----------
    parameter_path : str
        The path to the parameter in the model.
    distribution : DistributionBase
        The distribution defining the error for this parameter.
    """

    def __init__(self, parameter_path: str, distribution: DistributionBase):
        """
        Initialize the VariatedVariable.

        Parameters
        ----------
        parameter_path : str
            The path to the parameter in the model.
        distribution : DistributionBase
            The distribution defining the error for this parameter.

        Raises
        ------
        TypeError
            If distribution is not an instance of DistributionBase.
        """
        if not isinstance(distribution, DistributionBase):
            raise TypeError(
                f"Expected distribution to be an instance of DistributionBase, got {type(distribution).__name__}."
            )
        self.parameter_path = parameter_path
        self.distribution = distribution

    def __repr__(self):
        return f"VariatedVariable(parameter_path='{self.parameter_path}', distribution={self.distribution})"
