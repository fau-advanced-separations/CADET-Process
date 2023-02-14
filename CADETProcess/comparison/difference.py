from abc import abstractmethod
import copy
from functools import wraps
from warnings import warn

import numpy as np
from scipy.integrate import simps
from scipy.special import expit

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import UnsignedInteger
from CADETProcess.solution import SolutionBase, slice_solution
from CADETProcess.metric import MetricBase
from .shape import pearson, pearson_offset
from .peaks import find_peaks, find_breakthroughs


__all__ = [
    'DifferenceBase',
    'calculate_sse', 'calculate_rmse',
    'SSE', 'RMSE', 'NRMSE',
    'Norm', 'L1', 'L2',
    'AbsoluteArea', 'RelativeArea',
    'Shape',
    'PeakHeight', 'PeakPosition',
    'BreakthroughHeight', 'BreakthroughPosition',
]


def squishify(*args, **kwargs):
    warn(
        'This function is deprecated, use sigmoid_distance.',
        DeprecationWarning, stacklevel=2
    )
    return sigmoid_distance(*args, **kwargs)


def sigmoid_distance(measurement, target, normalization=1):
    """Calculate the distance between two values using a sigmoid function.

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
    reference : ReferenceIO
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

    def __init__(
            self,
            reference,
            components=None,
            use_total_concentration=False,
            use_total_concentration_components=True,
            start=None,
            end=None,
            transform=None,
            resample=True,
            smooth=False,
            normalize=False):
        """Initialize an instance of DifferenceBase.

        Parameters
        ----------
        reference : ReferenceIO
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
        resample : bool, optional
            If True, resample data. The default is True.
        smooth : bool, optional
            If True, smooth data. The default is False.
        normalize : bool, optional
            If True, normalize data. The default is False.

        """
        self.components = components
        self.use_total_concentration = use_total_concentration
        self.use_total_concentration_components = \
            use_total_concentration_components
        self.start = start
        self.end = end
        self.transform = transform
        self.resample = resample
        self.smooth = smooth
        self.normalize = normalize
        self.reference = reference

    @property
    def n_metrics(self):
        """int: Number of metrics."""
        return self.reference.solution.shape[-1]

    @property
    def bad_metrics(self):
        """list: Worst case values for each metric."""
        return self.n_metrics * [np.inf]

    @property
    def reference(self):
        """SolutionBase: The reference Solution, sliced and transformed."""
        return self._reference_sliced_and_transformed

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, SolutionBase):
            raise TypeError("Expected SolutionBase")

        self._reference = copy.deepcopy(reference)
        if self.resample and not self._reference.is_resampled:
            self._reference.resample()
        if self.normalize and not self._reference.is_normalized:
            self._reference.normalize()
        if self.smooth and not self._reference.is_smoothed:
            self._reference.smooth_data()
        reference = slice_solution(
            self._reference,
            None,
            self.use_total_concentration,
            self.use_total_concentration_components,
            coordinates={'time': (self.start, self.end)}
        )

        self._reference_sliced_and_transformed = reference

    def checks_dimensions(func):
        """Decorator to automatically check reference and solution dimensions."""
        @wraps(func)
        def wrapper(self, solution, *args, **kwargs):
            if solution.solution.shape != self.reference.solution.shape:
                raise CADETProcessError(
                    "Simulation shape does not match experiment"
                )
            value = func(self, solution, *args, **kwargs)
            return value
        return wrapper

    def slices_solution(func):
        """Decorator to automatically slice solution."""
        @wraps(func)
        def wrapper(self, solution, slice=True, *args, **kwargs):
            if slice:
                solution = slice_solution(
                    solution,
                    self.components,
                    self.use_total_concentration,
                    self.use_total_concentration_components,
                    coordinates={'time': (self.start, self.end)}
                )

            value = func(self, solution, *args, **kwargs)

            return value

        return wrapper

    def resamples_smoothes_and_normalizes_solution(func):
        """Decorator to automatically smooth and normalize solution."""
        @wraps(func)
        def wrapper(self, solution, *args, **kwargs):
            solution = copy.deepcopy(solution)
            solution.resample(
                self._reference.time[0],
                self._reference.time[-1],
                len(self._reference.time),
            )
            if self.normalize and not solution.is_normalized:
                solution.normalize()
            if self.smooth and not solution.is_smoothed:
                solution.smooth_data()

            value = func(self, solution, *args, **kwargs)
            return value
        return wrapper

    def transforms_solution(func):
        """Decorator to automatically transform solution data."""
        @wraps(func)
        def wrapper(self, solution, *args, **kwargs):
            if self.transform is not None:
                solution = copy.deepcopy(solution)
                solution.solution = self.transform(solution.solution)

            value = func(self, solution, *args, **kwargs)
            return value
        return wrapper

    @abstractmethod
    def _evaluate(self):
        """Abstract method to compute the difference metric.

        Returns
        -------
        np.ndarray
            The computed difference metric.
        """
        return

    @resamples_smoothes_and_normalizes_solution
    @slices_solution
    @transforms_solution
    @checks_dimensions
    def evaluate(self, solution):
        """Compute the difference between the reference solution and the input solution.

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

    @resamples_smoothes_and_normalizes_solution
    @slices_solution
    @transforms_solution
    def slice_and_transform(self, solution):
        """Slice the solution and applies the transform callable (if defined).

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


def calculate_sse(simulation, reference):
    """Calculate the sum of squared errors (SSE) between simulation and reference.

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

    def _evaluate(self, solution):
        sse = calculate_sse(solution.solution, self.reference.solution)

        return sse


def calculate_rmse(simulation, reference):
    """Calculate the root mean squared error (RMSE) between simulation and reference.

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

    def _evaluate(self, solution):
        rmse = calculate_rmse(solution.solution, self.reference.solution)

        return rmse


class NRMSE(DifferenceBase):
    """Normalized root mean squared errors (RRMSE) difference metric."""

    def _evaluate(self, solution):
        rmse = calculate_rmse(solution.solution, self.reference.solution)
        nrmse = rmse / np.max(self.reference.solution, axis=0)

        return nrmse


class Norm(DifferenceBase):
    """Norm difference metric.

    Attributes
    ----------
    order : int
        The order of the norm.
    """

    order = UnsignedInteger()

    def _evaluate(self, solution):
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

    def _evaluate(self, solution):
        """np.array: Absolute difference in area compared to reference.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        area_ref = simps(
            self.reference.solution, self.reference.time, axis=0
        )
        area_sol = simps(solution.solution, solution.time, axis=0)

        return abs(area_ref - area_sol)


class RelativeArea(DifferenceBase):
    """Relative difference in area difference metric."""

    def _evaluate(self, solution):
        """np.array: Relative difference in area compared to reference.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        area_ref = simps(
            self.reference.solution, self.reference.time, axis=0
        )
        area_new = simps(solution.solution, solution.time, axis=0)

        return abs(area_ref - area_new)/area_ref


class Shape(DifferenceBase):
    """Shape similarity difference metric.

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

    @wraps(DifferenceBase.__init__)
    def __init__(
            self, *args,
            use_derivative=True, normalize_metrics=True, normalization_factor=None,
            **kwargs):
        """Initialize Shape metric.

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

        self.peak_height = PeakHeight(
            *args, normalize=False, normalize_metrics=normalize_metrics, **kwargs
        )

        self.use_derivative = use_derivative
        if use_derivative:
            self.reference_der = self.reference.derivative
            self.reference_der.resample(
                start=self._reference.time[0],
                end=self._reference.time[-1],
                nt=len(self._reference.time)
            )
            self.reference_der_sliced = slice_solution(
                self.reference_der,
                None,
                self.use_total_concentration,
                self.use_total_concentration_components,
                coordinates={'time': (self.start, self.end)}
            )

            self.peak_der_min = PeakHeight(
                self.reference_der, *args[1:], normalize=False,
                find_minima=True, normalize_metrics=normalize_metrics,
                **kwargs
            )
            self.peak_der_max = PeakHeight(
                self.reference_der, *args[1:], normalize=False,
                find_minima=False, normalize_metrics=normalize_metrics,
                **kwargs
            )

        self.normalize_metrics = normalize_metrics
        if normalization_factor is None:
            normalization_factor = self.reference.time[-1]/10
        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self):
        if self.use_derivative:
            return 6
        else:
            return 3

    @property
    def labels(self):
        labels = ['Pearson Correleation', 'Time offset', 'Peak Height']
        if self.use_derivative:
            labels += [
                'Pearson Correlation Derivative',
                'Peak Minimum Derivative',
                'Peak Maximum Derivative'
                ]
        return labels

    def _evaluate(self, solution):
        """np.array: Shape similarity using pearson correlation.

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        corr, offset_original = pearson(
            self.reference.time,
            self.reference.solution_interpolated.solutions[0],
            solution.solution_interpolated.solutions[0],
        )

        peak_height = self.peak_height(solution, slice=False)

        if self.normalize_metrics:
            offset = sigmoid_distance(
                offset_original, target=0, normalization=self.normalization_factor
            )
        else:
            offset = np.abs(offset_original)

        if not self.use_derivative:
            return np.array([corr, offset, peak_height[0]])

        solution_der = solution.derivative
        solution_der_sliced = self.slice_and_transform(solution_der)

        corr_der = pearson_offset(
            self.reference_der_sliced.time,
            self.reference_der_sliced.solution_interpolated.solutions[0],
            solution_der_sliced.solution_interpolated.solutions[0],
            offset_original,
        )

        der_min = self.peak_der_min(solution_der_sliced, slice=False)
        der_max = self.peak_der_max(solution_der_sliced, slice=False)

        return np.array(
            [
                corr, offset, peak_height[0],
                corr_der, der_min[0], der_max[0]
            ]
        )


class PeakHeight(DifferenceBase):
    """Absolute difference in peak height difference metric.

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

    @wraps(DifferenceBase.__init__)
    def __init__(
            self, *args,
            find_minima=False, normalize_metrics=True, normalization_factor=None,
            **kwargs):
        """Initialize the PeakHeight object.

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
        self.reference_peaks = find_peaks(
            self.reference, find_minima=find_minima
        )

        self.normalize_metrics = normalize_metrics
        if normalization_factor is None:
            normalization_factor = [
                [peak[1] for peak in self.reference_peaks[i]]
                for i in range(self.reference.n_comp)
            ]
        else:
            normalization_factor = [
                len(self.reference_peaks[i])*[normalization_factor]
                for i in range(self.reference.n_comp)
            ]
        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self):
        return sum([len(peak) for peak in self.reference_peaks])

    def _evaluate(self, solution):
        """np.array: Difference in peak height (concentration).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        solution_peaks = find_peaks(solution, find_minima=self.find_minima)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[1], ref[1], factor)
                for i in range(self.reference.n_comp)
                for ref, sol, factor in zip(
                     self.reference_peaks[i],
                     solution_peaks[i],
                     self.normalization_factor
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
    """Absolute difference in peak peak position difference metric.

    Attributes
    ----------
    reference_peaks : list of tuple
        Contains the peaks found in the reference.
    normalize_metrics : bool
        Indicates whether normalization is applied to the peak height difference scores.
    normalization_factor : list of list
        Contains the normalization factors for each peak in each component.
    """

    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize_metrics=True, normalization_factor=None, **kwargs):
        """Initialize PeakPosition object.

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
                len(self.reference_peaks[i])*[normalization_factor]
                for i in range(self.reference.n_comp)
            ]

        # if normalization_factor is None:
        #     normalization_factor = self.reference.time[-1]/10
        self.normalization_factor = normalization_factor

    @property
    def n_metrics(self):
        return sum([len(peak) for peak in self.reference_peaks])

    def _evaluate(self, solution):
        """np.array: Difference in peak position (time).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        solution_peaks = find_peaks(solution)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[0], ref[0], factor)
                for i in range(self.reference.n_comp)
                for ref, sol, factor in zip(
                    self.reference_peaks[i],
                    solution_peaks[i],
                    self.normalization_factor[i]
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
    """Absolute difference in breakthrough curve height difference metric.

    Attributes
    ----------
    normalize_metrics : bool
        Whether to normalize the difference scores based on a sigmoid function.
    reference_bt : list of tuple
        List of breakthrough curves in the reference solution.

    """

    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize_metrics=True, **kwargs):
        """Initialize BreakthroughHeight metric.

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
    def n_metrics(self):
        return len(self.reference_bt)

    def _evaluate(self, solution):
        """np.array: Difference in breakthrough height (concentration).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        solution_bt = find_breakthroughs(solution)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[1], ref[1])
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [
                ref[1] - sol[1]
                for ref, sol in zip(self.solution_bt, solution_bt)
            ]

        return np.abs(score)


class BreakthroughPosition(DifferenceBase):
    """Absolute difference in breakthrough curve position difference metric."""

    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize_metrics=True, normalization_factor=None, **kwargs):
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
    def n_metrics(self):
        return len(self.reference_bt)

    def _evaluate(self, solution):
        """np.array: Difference in breakthrough position (time).

        Parameters
        ----------
        solution : SolutionIO
            Concentration profile of simulation.

        """
        solution_bt = find_breakthroughs(solution)

        if self.normalize_metrics:
            score = [
                sigmoid_distance(sol[0], ref[0], self.normalization_factor)
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [
                ref[0] - sol[0]
                for ref, sol in zip(self.solution_bt, solution_bt)
            ]

        return np.abs(score)
