"""
=========================================
Smoothing (:mod:`CADETProcess.smoothing`)
=========================================

.. currentmodule:: CADETProcess.smoothing

This module provides functionality for smoothing data.


.. autosummary::
    :toctree: generated/

    find_smoothing_factors
    full_smooth

"""

import multiprocessing

import numpy as np
import scipy.signal

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize

butter_order = 3

__all__ = ['find_smoothing_factors', 'full_smooth']


class TargetProblem(ElementwiseProblem):
    def __init__(self, lb, ub, sse_target, func, values, fs):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=lb, xu=ub)
        self.sse_target = sse_target
        self.func = func
        self.values = values
        self.fs = fs

    def _evaluate(self, crit_fs, out, *args, **kwargs):
        crit_fs = 10**crit_fs
        try:
            sos = self.func(crit_fs, self.fs)
            low_passed = scipy.signal.sosfiltfilt(sos, self.values)
            sse = np.sum((low_passed - self.values) ** 2)

            error = (sse - self.sse_target)**2
        except (ValueError, np.linalg.LinAlgError):
            error = np.inf
        out["F"] = error


class MaxDistance(ElementwiseProblem):
    def __init__(self, lb, ub, func, fs, values, x_min, y_min, p1, p2, factor):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=lb, xu=ub)
        self.func = func
        self.fs = fs
        self.values = values
        self.x_min = x_min
        self.y_min = y_min
        self.p1 = p1
        self.p2 = p2
        self.factor = factor

    def _evaluate(self, crit_fs, out, *args, **kwargs):
        crit_fs = 10.0 ** crit_fs[0]
        try:
            sos = self.func(crit_fs, self.fs)
        except ValueError:
            out['F'] = 1e6
            return

        try:
            low_passed = scipy.signal.sosfiltfilt(sos, self.values)
        except np.linalg.LinAlgError:
            out['F'] = 1e6
            return

        sse = np.sum((low_passed - self.values) ** 2)

        pT = np.array([crit_fs - self.x_min, np.log(sse) - self.y_min]).T / self.factor

        d = np.cross(self.p2 - self.p1, self.p1 - pT) / np.linalg.norm(self.p2 - self.p1)
        out["F"] = -d


def get_p(x, y):
    x = np.array(x)
    y = np.array(y)

    sort_idx = np.argsort(x)

    x = x[sort_idx]
    y = y[sort_idx]

    y_min = y - min(y)
    x_min = x - min(x)

    p3 = np.array([x_min, y_min]).T
    factor = np.max(p3, 0)
    p3 = p3 / factor
    p1 = p3[0, :]
    p2 = p3[-1, :]

    return x, min(x), y, min(y), p1, p2, p3, factor


def signal_bessel(crit_fs, fs):
    return scipy.signal.bessel(
        butter_order, crit_fs, btype="lowpass", analog=False, fs=fs,
        output="sos", norm="delay"
    )


def signal_butter(crit_fs, fs):
    return scipy.signal.butter(
        butter_order, crit_fs, btype="lowpass", analog=False, fs=fs,
        output="sos"
    )


def refine_signal(func, times, values, x, y, fs, start):
    x, x_min, y, y_min, p1, p2, p3, factor = get_p(x, y)

    lb = np.log10(x[0])
    ub = np.log10(x[-1])

    problem = MaxDistance(
        lb, ub, func, fs, values, x_min, y_min, p1, p2, factor
    )

    algorithm = PatternSearch(n_sample_points=50, eps=1e-13)

    res = minimize(
        problem,
        algorithm,
        verbose=False,
        seed=1
    )

    crit_fs = 10**res.X[0]

    return crit_fs


def find_L(x, y):
    # find the largest value greater than 0
    # otherwise return none to just turn off butter filter
    x, x_min, y, y_min, p1, p2, p3, factor = get_p(x, y)

    d = np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1)

    max_idx = np.argmax(d)
    max_d = d[max_idx]
    l_x = x[max_idx]
    l_y = y[max_idx]

    if max_d <= 0:
        return None, None

    return l_x, l_y


def find_signal(func, times, values, sse_target):
    filters = []
    sse = []

    fs = 1.0 / (times[1] - times[0])

    ub = fs / 2.0

    for i in np.logspace(-8, np.log10(ub), 50):
        try:
            sos = func(i, fs)
            low_passed = scipy.signal.sosfiltfilt(sos, values)

            filters.append(i)
            sse.append(np.sum((low_passed - values) ** 2))
        except (ValueError, np.linalg.LinAlgError):
            continue

    crit_fs_max = find_max_signal(
        func, times, values, sse_target, filters, sse
    )

    L_x, L_y = find_L(filters, np.log(sse))

    if L_x is not None:
        L_x = refine_signal(
            func, times, values, filters, np.log(sse), fs, L_x
        )

    if L_x is not None and crit_fs_max < L_x:
        L_x = crit_fs_max

    return L_x


def find_max_signal(func, times, values, sse_target, filters, sse):
    fs = 1.0 / (times[1] - times[0])

    filters = np.log10(filters)
    problem = TargetProblem(
        filters[0], filters[-1], sse_target, func, values, fs
    )

    algorithm = PatternSearch(n_sample_points=50, eps=1e-13)

    res = minimize(
        problem,
        algorithm,
        verbose=False,
        seed=1
    )

    crit_fs = 10**res.X[0]

    return crit_fs


def smoothing_filter_signal(func, times, values, crit_fs):
    if crit_fs is None:
        return values
    fs = 1.0 / (times[1] - times[0])
    sos = func(crit_fs, fs)
    low_passed = scipy.signal.sosfiltfilt(sos, values)
    return low_passed


def find_smoothing_factors(times, values, rmse_target=1e-4):
    sse_target = (rmse_target**2.0)*len(values)

    try:
        crit_fs = find_signal(signal_bessel, times, values, sse_target)
    except np.linalg.LinAlgError:
        crit_fs = None

    if crit_fs is None:
        multiprocessing.get_logger().info(
            "butter filter disabled, no viable L point found"
        )

    values_filter = smoothing_filter_signal(
        signal_bessel, times, values, crit_fs
    )

    s = sse_target

    spline, factor = create_spline(times, values, crit_fs, s)

    # run a quick butter pass to remove high frequency noise in the derivative
    # (needed for some experimental data)
    values_filter = spline.derivative()(times) / factor
    factor = 1.0/np.max(values_filter)
    values_filter = values_filter * factor
    crit_fs_der = find_signal(signal_bessel, times, values_filter, sse_target)

    return s, crit_fs, crit_fs_der


def create_spline(times, values, crit_fs, s):
    factor = 1.0 / np.max(values)
    values = values * factor
    values_filter = smoothing_filter_signal(
        signal_bessel, times, values, crit_fs
    )

    return (
        scipy.interpolate.UnivariateSpline(
            times, values_filter, s=s, k=5, ext=3
        ),
        factor,
    )


def smooth_data(times, values, crit_fs, s):
    spline, factor = create_spline(times, values, crit_fs, s)

    return spline(times) / factor


def smooth_data_derivative(
        times, values, crit_fs, s, crit_fs_der, smooth=True):
    spline, factor = create_spline(times, values, crit_fs, s)

    if smooth:
        values_filter_der = spline.derivative()(times) / factor

        factor_der = 1.0/np.max(values_filter_der)
        values_filter_der = values_filter_der * factor_der
        values_filter_der = butter(
            times, values_filter_der, crit_fs_der
        )
        values_filter_der = values_filter_der / factor_der
        spline_der = scipy.interpolate.InterpolatedUnivariateSpline(
            times, values_filter_der, k=5, ext=3
        )
        values_filter_der = spline_der(times)
    else:
        values_filter_der = spline.derivative()(times) / factor
    return values_filter_der


def full_smooth(times, values, crit_fs, s, crit_fs_der, smooth=True):
    # return smooth data derivative of data
    spline, factor = create_spline(times, values, crit_fs, s)

    values_filter = spline(times) / factor

    # run a quick butter pass to remove high frequency noise in the derivative
    # (needed for some experimental data)

    if smooth:
        values_filter_der = spline.derivative()(times) / factor

        factor_der = 1.0/np.max(values_filter_der)
        values_filter_der = values_filter_der * factor_der
        values_filter_der = butter(
            times, values_filter_der, crit_fs_der
        )
        values_filter_der = values_filter_der / factor_der
        spline_der = scipy.interpolate.InterpolatedUnivariateSpline(
            times, values_filter_der, k=5, ext=3
        )
        values_filter_der = spline_der(times)
    else:
        values_filter_der = spline.derivative()(times) / factor
    return values_filter, values_filter_der


def butter(times, values, crit_fs_der):
    values_filter = \
        smoothing_filter_signal(signal_bessel, times, values, crit_fs_der)

    return values_filter
