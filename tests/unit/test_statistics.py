import numpy as np

from thermal_optimal_path.statistics import average_path
from thermal_optimal_path.lattice import partition_function


def test_averaged_path_no_lag():
    """ Tests that for identical series input, there is no estimated lag according to the
    average path, ie it is zero at all times.
    """
    n = 10
    a = range(n)
    b = range(n)
    temperature = 1
    g = partition_function(a, b, temperature)
    avg = average_path(g)
    assert len(avg) == 2 * n - 1
    assert not np.isnan(avg).any()
    assert np.isfinite(g).all()
    # exclude values set by boundary conditions
    assert np.allclose(avg, 0)
