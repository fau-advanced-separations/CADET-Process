from abc import abstractmethod
import copy
from functools import wraps

import numpy as np
from scipy.integrate import simps
from scipy.special import expit

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import UnsignedInteger
from CADETProcess.processModel import ComponentSystem
from CADETProcess.solution import SolutionBase
from CADETProcess.metric import MetricBase
from .shape import pearson, pearson_offset
from .peaks import find_peaks, find_breakthroughs


def squishify(measurement, target, normalization=1):
    input = (measurement - target)/normalization
    output = np.abs(2*expit(input)-1)
    return output


def slice_solution(
        solution_original,
        components=None,
        use_total_concentration=False, use_total_concentration_components=True,
        start=0, end=None):
    solution = copy.deepcopy(solution_original)

    start_index = np.where(solution.time >= start)[0][0]
    if end is not None:
        end_index = np.where(solution.time > end)[0][0]
    else:
        end_index = None

    solution.time = solution.time[start_index:end_index]
    solution.solution = solution.solution[start_index:end_index, ...]

    if components is not None:
        component_system = copy.deepcopy(solution.component_system)
        component_indices = []
        for i, (name, component) in enumerate(
                solution.component_system.components_dict.items()):
            if name not in components:
                component_system.remove_component(component.name)
            else:
                if use_total_concentration_components:
                    component_indices.append(i)
                else:
                    component_indices.append(
                        component_system.indices[component.name]
                    )

        if use_total_concentration_components:
            solution.component_system = component_system
            solution.solution = \
                solution_original.total_concentration_components[start_index:end_index, ..., component_indices]
        else:
            solution.solution = \
                solution_original.solution[start_index:end_index, ..., component_indices]

    if use_total_concentration:
        solution_comp = copy.deepcopy(solution)
        solution.component_system = ComponentSystem(1)
        solution.solution = np.array(solution_comp.total_concentration, ndmin=2).transpose()

    return solution


def sse(simulation, reference):
    return np.sum((simulation - reference) ** 2, axis=0)


class DifferenceBase(MetricBase):
    def __init__(
            self,
            reference,
            components=None,
            reference_component_index=None,
            use_total_concentration=False,
            use_total_concentration_components=True,
            start=0,
            end=None,
            transform=None):
        self.reference = reference
        self.reference_component_index = reference_component_index
        self.components = components
        self.use_total_concentration = use_total_concentration
        self.use_total_concentration_components = \
            use_total_concentration_components
        self.start = start
        self.end = end
        self.transform = transform

    @property
    def n_metrics(self):
        return self.reference.solution.shape[-1]

    @property
    def bad_metrics(self):
        return self.n_metrics * [np.inf]

    @property
    def reference(self):
        return slice_solution(
            self._reference, self.reference_component_index,
            use_total_concentration=self.use_total_concentration,
            use_total_concentration_components=self.use_total_concentration_components,
            start=self.start, end=self.end
        )

    @reference.setter
    def reference(self, reference):
        if not isinstance(reference, SolutionBase):
            raise TypeError("Expected SolutionBase")

        self._reference = reference

    def checks_dimensions(func):
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

    def transforms_solution(func):
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

    @slices_solution
    @transforms_solution
    @checks_dimensions
    def evaluate(self, solution):
        metric = self._evaluate(solution)
        return metric

    __call__ = evaluate

    @slices_solution
    @transforms_solution
    def slice_and_transform(self, solution):
        return solution


class SSE(DifferenceBase):
    def _evaluate(self, solution):
        return sse(solution.solution, self.reference.solution)


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
        area_ref = simps(self.reference.solution, self.reference.time, axis=0)
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
        area_ref = simps(self.reference.solution, self.reference.time, axis=0)
        area_new = simps(solution.solution, solution.time, axis=0)

        return abs(area_ref - area_new)/area_ref


class Shape(DifferenceBase):
    use_derivative = True

    @wraps(DifferenceBase.__init__)
    def __init__(
            self, *args,
            use_derivative=0, normalize=True, normalization=None,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.peak_height = PeakHeight(*args, normalize=normalize, **kwargs)

        self.use_derivative = use_derivative
        if use_derivative:
            self.reference_der = copy.deepcopy(self.reference)
            self.reference_der.time_original = self.reference.time
            der_fun = self.reference_der.solution_interpolated.derivative
            self.reference_der.solution_original = der_fun(self.reference.time)
            self.reference_der.reset()

            self.peak_der_min = PeakHeight(
                self.reference_der, *args[1:],
                find_minima=True, normalize=normalize,
                **kwargs
            )
            self.peak_der_max = PeakHeight(
                self.reference_der, *args[1:],
                find_minima=False, normalize=normalize,
                **kwargs
            )

        self.normalize = normalize
        if normalization is None:
            normalization = self.reference.time[-1]/10
        self.normalization = normalization

    @property
    def n_metrics(self):
        if self.use_derivative:
            return 6
        else:
            return 3

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

        if self.normalize:
            offset = squishify(
                offset_original, target=0, normalization=self.normalization
            )
        else:
            offset = offset_original

        if not self.use_derivative:
            return np.array([corr, offset, peak_height[0][0]])

        solution_der = copy.deepcopy(solution)
        solution_der.time_original = self.reference.time
        der_fun = solution_der.solution_interpolated.derivative
        solution_der.solution_original = der_fun(self.reference.time)
        solution_der.reset()

        corr_der = pearson_offset(
            self.reference.time,
            self.reference_der.solution_interpolated.solutions[0],
            solution_der.solution_interpolated.solutions[0],
            offset_original,
        )

        der_min = self.peak_der_min(solution_der, slice=False)
        der_max = self.peak_der_max(solution_der, slice=False)

        return np.array(
            [
                corr, offset, peak_height[0][0],
                corr_der, der_min[0][0], der_max[0][0]
            ]
        )


class PeakHeight(DifferenceBase):
    @wraps(DifferenceBase.__init__)
    def __init__(
            self, *args,
            find_minima=False, normalize=True, normalization=None,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.find_minima = find_minima
        self.reference_peaks = find_peaks(
            self.reference, find_minima=find_minima
        )
        self.normalize = normalize
        if normalization is None:
            normalization = [
                [peak[1] for peak in self.reference_peaks[i]]
                for i in range(self.reference.n_comp)
            ]
        else:
            normalization = [
                len(self.reference_peaks[i])*[normalization]
                for i in range(self.reference.n_comp)
            ]
        self.normalization = normalization

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

        if self.normalize:
            score = [
                squishify(sol[1], ref[1], norm)
                for i in range(self.reference.n_comp)
                for ref, sol, norm in zip(self.reference_peaks[i], solution_peaks[i], self.normalization)
            ]
        else:
            score = [
                ref[1] - sol[1]
                for i in range(self.reference.n_comp)
                for ref, sol in zip(self.reference_peaks[i], solution_peaks[i])
            ]

        return np.abs(score)


class PeakPosition(DifferenceBase):
    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize=True, normalization=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.reference_peaks = find_peaks(self.reference)
        self.normalize = normalize

        if normalization is None:
            normalization = self.reference.time[-1]/10
        self.normalization = normalization

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

        if self.normalize:
            score = [
                squishify(sol[0], ref[0], std[i])
                for i in range(self.reference.n_comp)
                for ref, sol, std in zip(
                    self.reference_peaks[i], solution_peaks[i], self.std
                )
            ]
        else:
            score = [
                ref[0] - sol[0]
                for i in range(self.reference.n_comp)
                for ref, sol in zip(self.reference_peaks[i], solution_peaks[i])
            ]

        return np.abs(score)


class BreakthroughHeight(DifferenceBase):
    @wraps(DifferenceBase.__init__)
    def __init__(self, *args, normalize=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.reference_bt = find_breakthroughs(self.reference)
        self.normalize = normalize

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

        if self.normalize:
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
    def __init__(self, *args, normalize=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.reference_bt = find_breakthroughs(self.reference)
        self.normalize = normalize

        t_min = self.reference.time[0]
        t_max = self.reference.time[-1]
        self.std = max(
            self.reference_bt[0] - t_min, t_max - self.reference_bt[0]
        )

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

        if self.normalize:
            score = [
                squishify(sol[0], ref[0], self.std)
                for ref, sol in zip(self.reference_bt, solution_bt)
            ]
        else:
            score = [
                ref[0] - sol[0]
                for ref, sol in zip(self.solution_bt, solution_bt)
            ]

        return np.abs(score)
