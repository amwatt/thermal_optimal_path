import numpy as np
from numba import jit

from thermal_optimal_path.lattice import iter_lattice


@jit
def average_path(partition_function):
    """ Computes the average path of the partition function. The average includes the values
    set at the boundaries.

    The path is of length 2n-1 where n is the length of the original series, as the path
    is in the new time coordinate. To get a path of length n with the same interpretation as
    a series in the original coordinates, iterate over every other value of the result.
    """
    n = partition_function.shape[0]
    total = np.empty(2*n-1)
    total.fill(np.nan)
    avg = np.zeros(2*n-1)
    for x, t, t_a, t_b in iter_lattice(n, exclude_boundary=False):
        if np.isnan(total[t]):
            total[t] = 0
        total[t] += partition_function[t_a, t_b]
        avg[t] += x * partition_function[t_a, t_b]
    return avg / total
