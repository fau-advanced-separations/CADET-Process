from __future__ import annotations

import copy
from abc import abstractmethod
from functools import wraps
from typing import Any, Callable, Optional, Union

import numpy as np
from scipy.integrate import simpson
from scipy.special import expit

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import UnsignedInteger
from CADETProcess.metric import MetricBase
from CADETProcess.reference import FractionationReference, ReferenceBase, ReferenceIO
from CADETProcess.solution import (
    SolutionBase,
    SolutionIO,
    slice_solution,
    slice_solution_front,
)

from .peaks import find_breakthroughs, find_peaks
from .shape import pearson, pearson_offset

__all__ = [
    "DifferenceBase",
    "calculate_sse",
    "calculate_rmse",
    "SSE",
    "RMSE",
    "NRMSE",
    "Norm",
    "L1",
    "L2",
    "AbsoluteArea",
    "RelativeArea",
    "Shape",
    "ShapeFront",
    "PeakHeight",
    "PeakPosition",
    "BreakthroughHeight",
    "BreakthroughPosition",
    "FractionationSSE",
]


def sigmoid_distance(
    measurement: Union[float, np.ndarray],
    target: Union[float, np.ndarray],
    normalization: float = 1,
) -> float | np.ndarray:
    """
    Calculate the distance between two values using a sigmoid function.

    The distance is defined as the sigmoid transformation of the difference between the
    measurement and target values, normalized by a user-specified factor.

    Parameters
    ----------
    measurement : float or numpy.ndarray
        The measurement value(s).
    target : float or numpy.ndarray
        The target value(s).
    normalization : float, optional
        The factor to use for normalization. Default is 1.

    Returns
    -------
    float or numpy.ndarray
        The sigmoid distance between the measurement and target values.

    Examples
    --------
    >>> sigmoid_distance(3, 5)
    0.7310585786300049

    >>> sigmoid_distance([1, 2, 3], [3, 2, 1])
    array([0.73105858, 0.5       , 0.26894142])
    """
    input = (measurement - target) / normalization
    output = np.abs(2 * expit(input) - 1)
    return output


class DifferenceBase(MetricBase):
    """
    Base class for difference metric evaluation between a reference and a solution.

    Parameters
    ----------
    reference : ReferenceBase
        Reference used for calculating difference metric.
    components : {str, list}, optional
        Solution components to be considered.
        If None, all components are considered. The default is None.
    use_total_concentration : bool, optional
        If True, use sum of all components. The default is False.
    use_total_concentration_components : bool, optional
        If True, sum concentration of species. The default is True.
    start : float, optional
        End time of solution slice to be considered. The default is None.
    end : float, optional
        End time of solution slice to be considered. The default is None.
    transform : callable, optional
        Function to transform solution. The default is None.
    resample : bool, optional
        If True, resample data. The default is True.
    smooth : bool, optional
        If True, smooth data. The default is False.
    normalize : bool, optional
        If True, normalize data. The default is False.
    """

    _valid_references: tuple[type] = ()

    def __init__(
        self,
        reference: ReferenceBase,
        components: Optional[dict[str, list]] = None,
        use_total_concentration: bool = False,
        use_total_concentration_components: bool = True,
        start: Optional[float] = None,
        end: Optional[float] = None,
        transform: Optional[Callable] = None,
        only_transforms_array: bool = True,
        resample: bool = True,
        smooth: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        Initialize an instance of DifferenceBase.

        Parameters
        ----------
        reference : ReferenceBase
            Reference used for calculating difference metric.
        components : {str, list}, optional
            Solution components to be considered.
            If None, all components are considered. The default is None.
        use_total_concentration : bool, optional
            If True, use sum of all components. The default is False.
        use_total_concentration_components : bool, optional
            If True, sum concentration of species. The default is True.
        start : float, optional
            End time of solution slice to be considerd. The default is None.
        end : float, optional
            End time of solution slice to be considerd. The default is None.
        transform : callable, optional
            Function to transform solution. The default is None.
        only_transforms_array: bool, optional
            If True, only transform np array of solution object. The default is True.
        resample : bool, optional
            If True, resample data. The default is True.
        smooth : bool, optional
            If True, smooth data. The default is False.
        normalize : bool, optional
            If True, normalize data. The default is False.
        """
        self.components = components
        self.use_total_concentration = use_total_concentration
        self.use_total_concentration_components = use_total_concentration_components
        self.start = start
        self.end = end
        self.transform = transform
        self.only_transforms_array = only_transforms_array
        self.resample = resample
        self.smooth = smooth
        self.normalize = normalize
        self.reference = reference

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics."""
        return self.reference.solution.shape[-1]

    @property
    def bad_metrics(self) -> list:
        """list: Worst case values for each metric."""
        return self.n_metrics * [np.inf]

    @property
    def reference(self) -> SolutionBase:
        """SolutionBase: The reference solution."""
        return self._reference

    @reference.setter
    def reference(self, reference: SolutionBase) -> None:
        if not isinstance(reference, self._valid_references):
            raise TypeError(
                f"Invalid reference type: {type(reference)}. "
                f"Expected types: {self._valid_references}."
            )

        reference = copy.deepcopy(reference)
        self.reference_original = reference

        if self.resample:
            reference = reference.resample()
        if self.normalize:
            reference = reference.normalize()
        if self.smooth:
            reference = reference.smooth_data()

        reference = slice_solution(
            reference,
            self.components,
            self.use_total_concentration,
            self.use_total_concentration_components,
            coordinates={"time": (self.start, self.end)},
        )

        self._reference = reference

    def checks_dimensions(func: Callable) -> Callable:
        """Wrap method automatically check reference and solution dimensions."""

        @wraps(func)
        def wrapper(
            self: DifferenceBase,
            solution: SolutionBase,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Automatically check reference and solution dimensions before further processing."""
            if solution.solution.shape != self.reference.solution.shape:
                raise CADETProcessError("Simulation shape does not match experiment")
            value = func(self, solution, *args, **kwargs)
            return value

        return wrapper

    def slices_solution(func: Callable) -> Callable:
        """Wrap method to automatically slice solution."""

        @wraps(func)
        def wrapper(
            self: "DifferenceBase",
            solution: SolutionBase,
            slice: bool = True,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Automatically slice solution before further processing."""
            if slice:
                solution = slice_solution(
                    solution,
                    self.components,
                    self.use_total_concentration,
                    self.use_total_concentration_components,
                    coordinates={"time": (self.start, self.end)},
                )

            value = func(self, solution, *args, **kwargs)

            return value

        return wrapper

    def resamples_normalizes_and_smoothes_solution(func: Callable) -> Callable:
        """Wrap method to automatically resample, normalize, and smooth solution."""

        @wraps(func)
        def wrapper(
            self: DifferenceBase,
            solution: SolutionBase,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Automatically resample, normalize, and smooth solution before further processing."""
            solution = copy.deepcopy(solution)
            if self.resample:
                solution = solution.resample(
                    self.reference.time[0],
                    self.reference.time[-1],
                    len(self.reference.time),
                )
            if self.normalize:
                solution = solution.normalize()
            if self.smooth:
                solution = solution.smooth_data()

            value = func(self, solution, *args, **kwargs)
            return value

        return wrapper

    def transforms_solution(func: Callable) -> Callable:
        """Wrap method s.t. solution is automatically transformed."""

        @wraps(func)
        def wrapper(
            self: DifferenceBase,
            solution: SolutionBase,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """Automatically transforme solution data before further processing."""
            if self.transform is not None:
                solution = copy.deepcopy(solution)

                if self.only_transforms_array:
                    solution.solution = self.transform(solution.solution)
                else:
                    solution = self.transform(solution)

            value = func(self, solution, *args, **kwargs)
            return value

        return wrapper

    @abstractmethod
    def _evaluate(self) -> np.ndarray:
        """
        Abstract method to compute the difference metric.

        Returns
        -------
        np.ndarray
            The computed difference metric.
        """
        return

    @resamples_normalizes_and_smoothes_solution
    @slices_solution
    @transforms_solution
    @checks_dimensions
    def evaluate(self, solution: SolutionBase) -> np.ndarray:
        """
        Compute the difference between the reference solution and the input solution.

        Parameters
        ----------
        solution : Solution
            The solution to compare with the reference solution.

        Returns
        -------
        np.ndarray
            The difference between the two solutions.
        """
        metric = self._evaluate(solution)
        return metric

    __call__ = evaluate

    @resamples_normalizes_and_smoothes_solution
    @slices_solution
    @transforms_solution
    def slice_and_transform(self, solution: SolutionBase) -> SolutionBase:
        """
        Slice the solution and applies the transform callable (if defined).

        Parameters
        ----------
        solution : Solution
            The solution to slice and transform.

        Returns
        -------
        Solution
            The sliced and transformed solution.
        """
        return solution


def calculate_sse(simulation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Calculate the sum of squared errors (SSE) between simulation and reference.

    Parameters
    ----------
    simulation : np.ndarray
        Array of simulated values.
    reference : np.ndarray
        Array of reference values.

    Returns
    -------
    np.ndarray
        The SSE between simulation and reference.
    """
    return np.sum((simulation - reference) ** 2, axis=0)


class SSE(DifferenceBase):
    """Sum of squared errors (SSE) difference metric."""

    _valid_references = (ReferenceIO, SolutionIO)

    def _evaluate(self, solution: SolutionBase) -> np.ndarray:
        sse = calculate_sse(solution.solution, self.reference.solution)

        return sse


def calculate_rmse(simulation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Calculate the root mean squared error (RMSE) between simulation and reference.

    Parameters
    ----------
    simulation : np.ndarray
        Array of simulated values.
    reference : np.ndarray
        Array of reference values.

    Returns
    -------
    np.ndarray
        The RMSE between simulation and reference.
    """
    return np.sqrt(np.mean((simulation - reference) ** 2, axis=0))


class RMSE(DifferenceBase):
    """Root mean squared errors (RMSE) difference metric."""

    _valid_references = (SolutionIO, ReferenceIO)

    def _evaluate(self, solution: SolutionBase) -> np.ndarray:
        rmse = calculate_rmse(solution.solution, self.reference.solution)

        return rmse


class NRMSE(DifferenceBase):
    """Normalized root mean squared errors (RRMSE) difference metric."""

    _valid_references = (SolutionIO, ReferenceIO)

    def _evaluate(self, solution: SolutionBase) -> np.ndarray:
        rmse = calculate_rmse(solution.solution, self.reference.solution)
        nrmse = rmse / np.max(self.reference.solution, axis=0)

        return nrmse


class Norm(DifferenceBase):
    """
    Norm difference metric.

    Attributes
    ----------
    order : int
        The order of the norm.
    """

    _valid_references = (SolutionIO, ReferenceIO)

    order = UnsignedInteger()

    def _evaluate(self, solution: SolutionBase) -> float:
        norm = np.linalg.norm(
            (solution.solution - self.reference.solution), ord=self.order
        )

        return norm


class L1(Norm):
    """L1 norm difference metric."""

    order = 1


class L2(Norm):
    """L2 norm difference metric."""

    order = 2


class AbsoluteArea(DifferenceBase):
    """Absolute difference in area difference metric."""

    _valid_references = (SolutionIO, ReferenceIO)

    def _evaluate(self, solution: SolutionBase) -> Union[float, np.ndarray]:
        """
        np.ndarray: Absolute difference in area compared to reference.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.
        """
        area_ref = simpson(self.reference.solution, self.reference.time, axis=0)
        area_sol = simpson(solution.solution, solution.time, axis=0)

        return abs(area_ref - area_sol)


class RelativeArea(DifferenceBase):
    """Relative difference in area difference metric."""

    _valid_references = (SolutionIO, ReferenceIO)

    def _evaluate(self, solution: SolutionBase) -> Union[np.ndarray, float]:
        """
        Calculate relative difference in area compared to reference.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray
            The relative difference in area comared to reference.
        """
        area_ref = simpson(self.reference.solution, x=self.reference.time, axis=0)
        area_new = simpson(solution.solution, x=solution.time, axis=0)

        return abs(area_ref - area_new) / area_ref


class Shape(DifferenceBase):
    """
    Shape similarity difference metric.

    The similarity is calculated using the Pearson correlation between the reference
    and solution profiles, as well as the time offset between their peak positions, and
    the peak height of the solution profile. Additionally, if `use_derivative` is set to
    True, the similarity is also calculated using the Pearson correlation of the derivative
    profiles, and the minimum and maximum peak heights of the derivative profile.

    Attributes
    ----------
    n_metrics : int
        Number of similarity metrics calculated by the class.
    labels : list of str
        List of labels for each similarity metric calculated by the class.

    Raises
    ------
    CADETProcessError
        If `components` is not None and has more than one element.

    Notes
    -----
    Currently, this class only works for single-component systems with one peak.
    """

    _valid_references = (SolutionIO, ReferenceIO)

    @wraps(DifferenceBase.__init__)
    def __init__(
        self,
        *args: Any,
        use_derivative: bool = True,
        normalize_metrics: bool = True,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Shape metric.

        Parameters
        ----------
        *args :
            Positional arguments for DifferenceBase.
        use_derivative : bool, optional
            If True, use the derivative profiles to calculate similarity.
            Default is True.
        normalize_metrics : bool, optional
            If True, normalize the similarity metrics to a range of [0, 1] using a
            sigmoid function.
            Default is True.
        normalization_factor : float, optional
            Normalization factor used by the sigmoid function.
            Default is None, which sets it to 1/10 of the simulation time.
        **kwargs : dict
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)

        if self.reference.n_comp > 1 and not self.use_total_concentration:
            if self.components is not None and len(self.components) == 1:
                pass
            else:
                raise CADETProcessError(
                    "Shape currently only supports single component."
                )

        self.use_derivative = use_derivative
        if use_derivative:
            self.reference_der = self.reference.derivative

        self.normalize_metrics = normalize_metrics
        if normalization_factor is None:
            normalization_factor = self.reference.time[-1] / 10
        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics."""
        n_metrics = 2

        if self.use_derivative:
            n_metrics += 1

        return n_metrics

    @property
    def labels(self) -> list:
        """list[str]: List of difference metric names."""
        labels = ["Pearson Correleation", "Time offset"]
        if self.use_derivative:
            labels += ["Pearson Correlation Derivative"]
        return labels

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Evaluate the Shape similarity using Pearson correlation.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray
            Array of similarity metrics.
        """
        corr, offset_original = pearson(
            self.reference.time,
            self.reference.solution_interpolated.solutions[0],
            solution.solution_interpolated.solutions[0],
        )

        if self.normalize_metrics:
            offset = sigmoid_distance(
                offset_original, target=0, normalization=self.normalization_factor
            )
        else:
            offset = np.abs(offset_original)

        if not self.use_derivative:
            return np.array([corr, offset])

        corr_der = pearson_offset(
            self.reference_der.time,
            self.reference_der.solution_interpolated.solutions[0],
            solution.derivative.solution_interpolated.solutions[0],
            offset_original,
        )

        return np.array(
            [
                corr,
                offset,
                corr_der,
            ]
        )


class ShapeFront(Shape):
    """
    ShapeFront similarity difference metric.

    This class extends the Shape metric by focusing on the front of a peak for
    similarity calculations.

    Attributes
    ----------
    n_metrics : int
        Number of similarity metrics calculated by the class.
    labels : list of str
        List of labels for each similarity metric calculated by the class.

    Notes
    -----
    This class is designed for single-component systems with one peak.
    """

    def __init__(
        self,
        *args: Any,
        min_percent: Optional[float] = 0.02,
        max_percent: Optional[float] = 0.98,
        use_max_slope: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ShapeFront metric.

        Parameters
        ----------
        *args
            Positional arguments for Shape.
        min_percent : Optional[float], default=0.02
            The minimum percentage value for the metric.
        max_percent : Optional[float], default=0.98
            The maximum percentage value for the metric.
        use_max_slope : Optional[bool], default=False
            If True, use the maximum slope in calculations.
        **kwargs : dict
            Keyword arguments for Shape.
        """
        super().__init__(*args, **kwargs)

        self.min_percent = min_percent
        self.max_percent = max_percent
        self.use_max_slope = use_max_slope

        self.reference_front, idx_min, idx_max = slice_solution_front(
            self.reference,
            self.min_percent,
            self.max_percent,
            use_max_slope=use_max_slope,
            return_indices=True,
        )

        if self.use_derivative:
            self.reference_der_front = copy.deepcopy(self.reference_der)
            self.reference_der_front.solution[:] = 0
            self.reference_der_front.solution[idx_min : idx_max + 1] = (
                self.reference_der.solution[idx_min : idx_max + 1]
            )
            self.reference_der_front.update_solution()

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Evaluate the ShapeFront similarity using Pearson correlation.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray
            Array of similarity metrics.
        """
        times = solution.time

        solution_front, idx_min, idx_max = slice_solution_front(
            solution,
            self.min_percent,
            self.max_percent,
            use_max_slope=self.use_max_slope,
            return_indices=True,
        )

        corr, offset_original = pearson(
            times,
            self.reference_front.solution_interpolated.solutions[0],
            solution_front.solution_interpolated.solutions[0],
        )

        if self.normalize_metrics:
            offset = sigmoid_distance(
                offset_original, target=0, normalization=self.normalization_factor
            )
        else:
            offset = np.abs(offset_original)

        if not self.use_derivative:
            return np.array([corr, offset])

        solution_der = solution.derivative

        solution_der_front = copy.deepcopy(solution_der)
        solution_der_front.solution[:] = 0
        solution_der_front.solution[idx_min : idx_max + 1] = (
                solution_der.solution[idx_min : idx_max + 1]
        )
        solution_der_front.update_solution()

        corr_der = pearson_offset(
            self.reference_der_front.time,
            self.reference_der_front.solution_interpolated.solutions[0],
            solution_der_front.solution_interpolated.solutions[0],
            offset_original,
        )

        return np.array(
            [
                corr,
                offset,
                corr_der,
            ]
        )


class PeakHeight(DifferenceBase):
    """
    Absolute difference in peak height difference metric.

    Attributes
    ----------
    find_minima : bool
        Indicates whether the minima instead of maxima of the peaks are found.
    reference_peaks : list of tuple
        Contains the peaks found in the reference.
    normalize_metrics : bool
        Indicates whether normalization is applied to the peak height difference scores.
    normalization_factor : list of list
        Contains the normalization factors for each peak in each component.
    """

    _valid_references = (SolutionIO, ReferenceIO)

    @wraps(DifferenceBase.__init__)
    def __init__(
        self,
        *args: Any,
        find_minima: bool = False,
        normalize_metrics: bool = True,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the PeakHeight object.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the base class constructor.
        find_minima : bool, optional
            If True, finds the minima instead of maxima of the peaks, by default False.
        normalize_metrics : bool, optional
            If True, applies normalization to the peak height difference scores.
            The default is True.
        normalization_factor : int or None, optional
            If not None, sets the normalization factor to a constant value for all peaks.
            If None, calculates the normalization factor based on the reference peaks
            The default is None.
        **kwargs : dict
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)

        self.find_minima = find_minima
        self.reference_peaks = find_peaks(self.reference, find_minima=find_minima)

        self.normalize_metrics = normalize_metrics
        if normalization_factor is None:
            normalization_factor = [
                [peak[1] for peak in self.reference_peaks[i]]
                for i in range(self.reference.n_comp)
            ]
        else:
            normalization_factor = [
                len(self.reference_peaks[i]) * [normalization_factor]
                for i in range(self.reference.n_comp)
            ]
        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics."""
        return sum([len(peak) for peak in self.reference_peaks])

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Calculate difference in peak height (concentration).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray
            Array of difference in peak height (concentration).
        """
        solution_peaks = find_peaks(solution, find_minima=self.find_minima)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[1], ref[1], factor)
                for i in range(self.reference.n_comp)
                for ref, sol, factor in zip(
                    self.reference_peaks[i],
                    solution_peaks[i],
                    self.normalization_factor,
                )
            ]
        else:
            score = [
                ref[1] - sol[1]
                for i in range(self.reference.n_comp)
                for ref, sol in zip(self.reference_peaks[i], solution_peaks[i])
            ]

        return np.abs(score).flatten()


class PeakPosition(DifferenceBase):
    """
    Absolute difference in peak peak position difference metric.

    Attributes
    ----------
    reference_peaks : list of tuple
        Contains the peaks found in the reference.
    normalize_metrics : bool
        Indicates whether normalization is applied to the peak height difference scores.
    normalization_factor : list of list
        Contains the normalization factors for each peak in each component.
    """

    _valid_references = (SolutionIO, ReferenceIO)

    @wraps(DifferenceBase.__init__)
    def __init__(
        self,
        *args: Any,
        normalize_metrics: bool = True,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PeakPosition object.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the parent class constructor.
        normalize_metrics : bool, optional
            Whether to normalize the difference metrics, by default True.
        normalization_factor : float or list of float, optional
            Normalization factor(s) to use for the difference metric(s). If a single
            float is given, it will be used for all the metrics. If a list of floats is
            given, it must have one element per component in the reference, by default
            None.
        **kwargs : dict
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)

        self.reference_peaks = find_peaks(self.reference)

        self.normalize_metrics = normalize_metrics
        if normalization_factor is None:
            normalization_factor = [
                [peak[0] for peak in self.reference_peaks[i]]
                for i in range(self.reference.n_comp)
            ]
        else:
            normalization_factor = [
                len(self.reference_peaks[i]) * [normalization_factor]
                for i in range(self.reference.n_comp)
            ]

        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics."""
        return sum([len(peak) for peak in self.reference_peaks])

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Calculate difference in peak position (time).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray
            The difference in peak position.
        """
        solution_peaks = find_peaks(solution)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[0], ref[0], factor)
                for i in range(self.reference.n_comp)
                for ref, sol, factor in zip(
                    self.reference_peaks[i],
                    solution_peaks[i],
                    self.normalization_factor[i],
                )
            ]
        else:
            score = [
                ref[0] - sol[0]
                for i in range(self.reference.n_comp)
                for ref, sol in zip(self.reference_peaks[i], solution_peaks[i])
            ]

        return np.abs(score).flatten()


class BreakthroughHeight(DifferenceBase):
    """
    Absolute difference in breakthrough curve height difference metric.

    Attributes
    ----------
    normalize_metrics : bool
        Whether to normalize the difference scores based on a sigmoid function.
    reference_bt : list of tuple
        List of breakthrough curves in the reference solution.
    """

    _valid_references = (SolutionIO, ReferenceIO)

    @wraps(DifferenceBase.__init__)
    def __init__(
        self, *args: Any, normalize_metrics: bool = True, **kwargs: Any
    ) -> None:
        """
        Initialize BreakthroughHeight metric.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the base class constructor.
        normalize_metrics : bool, optional
            Whether to normalize the difference scores based on a sigmoid function.
            Default is True.
        **kwargs : dict
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)

        self.normalize_metrics = normalize_metrics
        self.reference_bt = find_breakthroughs(self.reference)

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics."""
        return len(self.reference_bt)

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Calculate difference in breakthrough height (concentration).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray:
            The difference in breakthrough height.
        """
        solution_bt = find_breakthroughs(solution)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[1], ref[1])
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [ref[1] - sol[1] for ref, sol in zip(self.solution_bt, solution_bt)]

        return np.abs(score)


class BreakthroughPosition(DifferenceBase):
    """Absolute difference in breakthrough curve position difference metric."""

    _valid_references = (SolutionIO, ReferenceIO)

    @wraps(DifferenceBase.__init__)
    def __init__(
        self,
        *args: Any,
        normalize_metrics: bool = True,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the BreakthroughPosition object.

        Parameters
        ----------
        *args :
            Positional arguments for DifferenceBase.
        normalize_metrics : bool, optional
            Whether to normalize the metrics. Default is True.
        normalization_factor : float, optional
            Factor to use for normalization.
            If None, it is set to the maximum of the difference between the reference
            breakthrough and the start time, and the difference between the end time and
            the reference breakthrough.
        **kwargs : dict
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, **kwargs)

        self.reference_bt = find_breakthroughs(self.reference)

        t_min = self.reference.time[0]
        t_max = self.reference.time[-1]

        self.normalize_metrics = normalize_metrics
        if normalization_factor is None:
            normalization_factor = max(
                self.reference_bt[0] - t_min, t_max - self.reference_bt[0]
            )
        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self) -> int:
        """int: Number of metrics."""
        return len(self.reference_bt)

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Calculate difference in breakthrough position (time).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        Returns
        -------
        np.ndarray:
            The difference in breakthrough position.
        """
        solution_bt = find_breakthroughs(solution)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[0], ref[0], self.normalization_factor)
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [ref[0] - sol[0] for ref, sol in zip(self.solution_bt, solution_bt)]

        return np.abs(score)


class FractionationSSE(DifferenceBase):
    """Fractionation based score using SSE."""

    _valid_references = FractionationReference

    @wraps(DifferenceBase.__init__)
    def __init__(
        self,
        *args: Any,
        normalize_metrics: bool = True,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the FractionationSSE object.

        Parameters
        ----------
        *args :
            Positional arguments for DifferenceBase.
        normalize_metrics : bool, optional
            Whether to normalize the metrics. Default is True.
        normalization_factor : float, optional
            Factor to use for normalization.
            If None, it is set to the maximum of the difference between the reference
            breakthrough and the start time, and the difference between the end time and
            the reference breakthrough.
        **kwargs : dict
            Keyword arguments passed to the base class constructor.
        """
        super().__init__(*args, resample=False, only_transforms_array=False, **kwargs)

        if not isinstance(self.reference, FractionationReference):
            raise TypeError(
                "FractionationSSE can only work with FractionationReference"
            )

        def transform(solution: SolutionIO) -> SolutionIO:
            """Transform solution."""
            solution = copy.deepcopy(solution)
            solution_fractions = [
                solution.create_fraction(frac.start, frac.end)
                for frac in self.reference.fractions
            ]

            solution.time = np.array(
                [(frac.start + frac.end) / 2 for frac in solution_fractions]
            )
            solution.solution = np.array(
                [frac.concentration for frac in solution_fractions]
            )

            return solution

        self.transform = transform

    def _evaluate(self, solution: SolutionIO) -> np.ndarray:
        """
        Calculate the sum of squared errors between the solution and a reference solution.

        Parameters
        ----------
        solution : SolutionIO
            An object containing the concentration profile of the simulation to be evaluated.

        Returns
        -------
        np.ndarray
            The sum of squared errors (SSE) between the solution and the reference solution.
        """
        sse = calculate_sse(solution.solution, self.reference.solution)

        return sse
