from abc import abstractmethod
import copy
from functools import wraps

import numpy as np
from scipy.integrate import simps
from scipy.special import expit

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import UnsignedInteger
from CADETProcess.solution import SolutionBase, slice_solution
from CADETProcess.metric import MetricBase
from .shape import pearson, pearson_offset
from .peaks import find_peaks, find_breakthroughs


def squishify(measurement, target, normalization=1):
    input = (measurement - target)/normalization
    output = np.abs(2*expit(input)-1)
    return output


class DifferenceBase(MetricBase):
    def __init__(
            self,
            reference,
            components=None,
            use_total_concentration=False,
            use_total_concentration_components=True,
            start=0,
            end=None,
            transform=None,
            resample=True,
            smooth=False,
            normalize=False):
        """

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
            End time of solution slice to be considerd. The default is 0.
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
        self.reference = reference
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

    @property
    def n_metrics(self):
        return self.reference.solution.shape[-1]

    @property
    def bad_metrics(self):
        return self.n_metrics * [np.inf]

    @property
    def reference(self):
        if self._reference_sliced_and_transformed is None:
            if self.normalize and not self._reference.is_normalized:
                self._reference.normalize()
            if self.smooth and not self._reference.is_smoothed:
                self._reference.smooth_data()
            reference = slice_solution(
                self._reference,
                None,
                self.use_total_concentration,
                self.use_total_concentration_components,
                self.start, self.end,
            )

            self._reference_sliced_and_transformed = reference
        return self._reference_sliced_and_transformed

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, SolutionBase):
            raise TypeError("Expected SolutionBase")

        self._reference = copy.deepcopy(reference)
        self._reference_sliced_and_transformed = None

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
                    self.start, self.end,
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
        return

    @resamples_smoothes_and_normalizes_solution
    @slices_solution
    @transforms_solution
    @checks_dimensions
    def evaluate(self, solution):
        metric = self._evaluate(solution)
        return metric

    __call__ = evaluate

    @resamples_smoothes_and_normalizes_solution
    @slices_solution
    @transforms_solution
    def slice_and_transform(self, solution):
        return solution


def calculate_sse(simulation, reference):
    return np.sum((simulation - reference) ** 2, axis=0)


class SSE(DifferenceBase):
    def _evaluate(self, solution):
        sse = calculate_sse(solution.solution, self.reference.solution)

        return sse


def calculate_rmse(simulation, reference):
    return np.sqrt(np.mean((simulation - reference) ** 2, axis=0))


class RMSE(DifferenceBase):
    def _evaluate(self, solution):
        rmse = calculate_rmse(solution.solution, self.reference.solution)

        return rmse


class NRMSE(DifferenceBase):
    def _evaluate(self, solution):
        rmse = calculate_rmse(solution.solution, self.reference.solution)
        nrmse = rmse / np.max(solution.solution, axis=0)

        return nrmse


class Norm(DifferenceBase):
    order = UnsignedInteger()

    def _evaluate(self, solution):
        norm = np.linalg.norm(
            (solution.solution - self.reference.solution), ord=self.order
        )

        return norm


class L1(Norm):
    order = 1


class L2(Norm):
    order = 2


class AbsoluteArea(DifferenceBase):
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
    @wraps(DifferenceBase.__init__)
    def __init__(
            self, *args,
            use_derivative=True, normalize_metrics=True, normalization_factor=None,
            **kwargs):
        super().__init__(*args, **kwargs)

        if self.reference.n_comp > 1 and not self.use_total_concentration:
            if self.components is not None and len(self.components) == 1:
                pass
            else:
                raise CADETProcessError(
                    "Shape currently only supports single component."
                )

        self.peak_height = PeakHeight(
            *args, normalize=True, normalize_metrics=normalize_metrics, **kwargs
        )

        self.use_derivative = use_derivative
        if use_derivative:
            self.reference_der = self.reference.derivative
            self.reference_der.resample(
                start=self.reference.time[0],
                end=self.reference.time[-1],
                nt=len(self.reference.time)
            )
            self.reference_der_sliced = slice_solution(
                self.reference_der,
                None,
                self.use_total_concentration,
                self.use_total_concentration_components,
                self.start, self.end,
            )

            self.peak_der_min = PeakHeight(
                self.reference_der, *args[1:],
                find_minima=True, normalize_metrics=normalize_metrics,
                **kwargs
            )
            self.peak_der_max = PeakHeight(
                self.reference_der, *args[1:],
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

        Note
        ----
        Currently only works for single component with one peak.

        """
        corr, offset_original = pearson(
            self.reference.time,
            self.reference.solution_interpolated.solutions[0],
            solution.solution_interpolated.solutions[0],
        )

        peak_height = self.peak_height(solution, slice=False)

        if self.normalize_metrics:
            offset = squishify(
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
    @wraps(DifferenceBase.__init__)
    def __init__(
            self, *args,
            find_minima=False, normalize_metrics=True, normalization_factor=None,
            **kwargs):
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
                squishify(sol[1], ref[1], factor)
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
    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize_metrics=True, normalization_factor=None, **kwargs):
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
                squishify(sol[0], ref[0], factor)
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
    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize_metrics=True, **kwargs):
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
                squishify(sol[1], ref[1])
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [
                ref[1] - sol[1]
                for ref, sol in zip(self.solution_bt, solution_bt)
            ]

        return np.abs(score)


class BreakthroughPosition(DifferenceBase):
    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize_metrics=True, normalization_factor=None, **kwargs):
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
                squishify(sol[0], ref[0], self.normalization_factor)
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [
                ref[0] - sol[0]
                for ref, sol in zip(self.solution_bt, solution_bt)
            ]

        return np.abs(score)
