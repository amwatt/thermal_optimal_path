import numpy as np
from numba import jit

from thermal_optimal_path.error_models import error


@jit
def iter_lattice(n, exclude_boundary=True):
    """ Generator for the partition function integer coordinates in the following form:
    (space in the new coordinates, time in the new coordinates, original coord of the first axis,
    original coord of the second axis)

    Parameters
    ----------
    n: int
        The size of the
    exclude_boundary: bool, opt
        If True, the coordinates of the boundary are skipped.
    """
    start_time = 1 if exclude_boundary else 0
    for t in range(start_time, 2 * n):
        # determine [start, end] interval required for the coordinates to be
        # betwen 0 (incl) and n (excl)
        offset = 1 if exclude_boundary else 0
        start = max(t - 2 * n + 1, -t + offset)
        end = min(2 * n - t - 1, t - offset)
        # further restrict range so that coordinates need to be integers
        if (start + t) % 2:
            start += 1
        # increment by 2 so that coordinates are integers
        for x in range(start, end + 1, 2):
            t_a = (t - x) // 2
            t_b = (t + x) // 2
            yield x, t, t_a, t_b


def iter_lattice_brute_force(n, exclude_boundary=True):
    """ Alternative implementation of the lattice coordinates generator. Iterates over all
    latice values in a brute force manner. This is simpler but less computationally efficient and
    mainly provided for convenience for unit tests.
    """
    start = 1 if exclude_boundary else 0
    for t in range(2 * n):
        for x in range(-t, t):
            t_a = (t - x) / 2
            t_b = (t + x) / 2
            if t_a == int(t_a) and t_b == int(t_b) and start <= t_a < n and start <= t_b < n:
                yield x, t, int(t_a), int(t_b)


def partition_function(series_a, series_b, temperature, error_func=None):
    """ Computed the partition function given two time series and the temperature parameter.

    Parameters
    ----------
    series_a: array like
        The first series
    series_b: array like
        The second series
    temperature: positive number
        The temperature smoothing parameter. The higher the temperature, the more errors are
        discarded.
    error_func: function, optional
        Function returning the error given two floats. The first arg comes from series_a,
        the second from series_b. If not provided, uses a default error model.

    Returns
    -------
    A Numpy 2D array for the computed partition function.
    """
    if not error_func:
        error_func = error
    return _partition_function_impl(series_a, series_b, temperature, error_func)


@jit
def _partition_function_impl(series_a, series_b, temperature, error_func):
    """ Implementation of the partition function. This should run in jit's nopython mode.
    Please refer to the caller for more docstrings.
    """
    size_a = len(series_a)
    size_b = len(series_b)
    if size_a != size_b:
        raise NotImplementedError('Only series of same lengths are supported.')

    g = np.empty((size_a, size_b))
    g.fill(np.nan)

    # boundary conditions - prevent paths from remaining at the boundaries
    g[0, :] = 0
    g[:, 0] = 0
    g[0, 0] = 1
    g[0, 1] = 1
    g[1, 0] = 1

    for x, t, t_a, t_b in iter_lattice(size_a):
        g_sum = g[t_a, t_b - 1] + g[t_a - 1, t_b] + g[t_a - 1, t_b - 1]
        val = g_sum * np.exp(-error_func(series_a[t_a], series_b[t_b]) / temperature)
        g[t_a, t_b] = val
    return g
