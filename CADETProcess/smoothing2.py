import numpy as np
import scipy.signal

from CADETMatch import smoothing

from CADETProcess.comparison import calculate_sse


def smooth_butter(reference, cutoff=None, resample=True):
    reference.reset()
    reference.normalize()

    if resample:
        reference.resample()

    signal = reference.solution

    fs = 1.0 / (reference.time[1] - reference.time[0])
    nyquist = fs / 2    # 0.5 times the sampling frequency

    if cutoff is None:
        cutoff = nyquist/10

    sos = scipy.signal.butter(3, cutoff, output='sos')
    filtered_signal = scipy.signal.sosfiltfilt(sos, signal, axis=0)

    sse = calculate_sse(reference.solution, filtered_signal)

    reference.solution = filtered_signal
    reference.is_smoothed = True

    return sse


def smooth_savgol(reference, window_length=None, resample=True):
    reference.reset()
    reference.normalize()

    if resample:
        reference.resample()

    signal = reference.solution

    if window_length is None:
        window_length = len(signal)/10

    filtered_signal = scipy.signal.savgol_filter(
        signal, window_length, 3, mode='nearest', axis=0
    )

    sse = calculate_sse(reference.solution, filtered_signal)

    reference.solution = filtered_signal
    reference.is_smoothed = True

    return sse


def smooth_median(reference, kernel_size=None, resample=True):
    reference.reset()
    reference.normalize()

    if resample:
        reference.resample()

    signal = reference.solution

    if kernel_size is None:
        kernel_size = len(signal)/100

    kernel_size = round(kernel_size)

    if kernel_size % 2 == 0:
        kernel_size -= 1

    filtered_signal = np.zeros(reference.solution.shape)

    for i in range(reference.n_comp):
        filtered_signal[:, i] = scipy.signal.medfilt(signal[:, i], kernel_size)

    sse = calculate_sse(reference.solution, filtered_signal)

    reference.solution = filtered_signal
    reference.is_smoothed = True

    return sse


def smooth_cadet_match(reference):
    reference.reset()

    filtered_signal = np.zeros(reference.solution.shape)

    for i in range(reference.n_comp):

        signal = reference.solution[:, i]

        params = smoothing.find_smoothing_factors(reference.time, signal, None, None)
        filtered_signal[:, i], _ = smoothing.full_smooth(reference.time, signal, *params)

    reference.solution = filtered_signal
    reference.is_smoothed = True
