import pytest
import numpy as np

from thermal_optimal_path.lattice import iter_lattice, iter_lattice_brute_force, partition_function
from thermal_optimal_path.error_models import error


# test partition function values (first series input, second series input, output)
test_function_values = (
    ([1, 2], [2, 3], [[1, 1], [1, 3*np.exp(-error(2, 3))]]),
)

# test n=3 lattice coordinates iter with boundaries in order of appearance
test_coordinates_with_boundaries = [
    (0, 0, 0, 0),
    (-1, 1, 1, 0),
    (1, 1, 0, 1),
    (-2, 2, 2, 0),
    (0, 2, 1, 1),
    (2, 2, 0, 2),
    (-1, 3, 2, 1),
    (1, 3, 1, 2),
    (0, 4, 2, 2),
]

# test n=3 lattice coordinates iter without boundaries in order of appearance
test_coordinates_without_boundaries = [
    (0, 2, 1, 1),
    (-1, 3, 2, 1),
    (1, 3, 1, 2),
    (0, 4, 2, 2),
]

test_coordinates = (
    (False, test_coordinates_with_boundaries),
    (True, test_coordinates_without_boundaries)
)


@pytest.mark.parametrize("size, exclude_boundary, expected_count",
                         ((100, True, 99**2), (100, False, 100**2)))
def test_lattice_iter_dimension(size, exclude_boundary, expected_count):
    coordinates_covered = [coord for coord in iter_lattice(size, exclude_boundary=exclude_boundary)]
    # coordinates should be unique
    assert len(coordinates_covered) == len(set(coordinates_covered))
    # there should be n squared coordinates when including boundaries
    assert len(coordinates_covered) == expected_count
    # all coordinates should be integers
    assert all([type(c) is int for coords in coordinates_covered for c in coords])
    # time coordinate is from 0 to 2n-2, including boundaries
    assert max(c[1] for c in coordinates_covered) == 2*size-2
    # space coordinate is from -n+1 to n-1, including boundaries
    offset = 1 if exclude_boundary else 0
    assert max(c[0] for c in coordinates_covered) == size-1-offset


@pytest.mark.parametrize("exclude_boundary, expected_coords", test_coordinates)
def test_lattice_iter_values(exclude_boundary, expected_coords):
    n = 3
    coordinates_computed = [coord for coord in iter_lattice(n, exclude_boundary=exclude_boundary)]
    assert coordinates_computed == expected_coords


def test_lattice_alternative_implementation():
    """ Compare the lattice iterator to the brute force implementation.
    """
    n = 100
    coordinates_computed = [coord for coord in iter_lattice(n)]
    coordinates_expected = [coord for coord in iter_lattice_brute_force(n)]
    assert set(coordinates_computed) == set(coordinates_expected)


def test_partition_numeric_type():
    # Test that all values have been computed
    n = 100
    a = list(range(n))
    b = list(range(n))[::-1]
    temperature = 1
    func = partition_function(a, b, temperature)
    assert not np.isnan(func).any()
    assert np.isfinite(func).all()


def test_partition_symmetry():
    # Output should be symmetric when the inputs and boundary conditions are symmetric
    n = 100
    a = list(range(n))
    b = list(range(n))
    temperature = 1
    func = partition_function(a, b, temperature)
    assert np.allclose(func, func.T)


@pytest.mark.parametrize("a, b, expected", test_function_values)
def test_partition_values(a, b, expected):
    # Compare to a small example computed by hand
    temperature = 1
    func = partition_function(a, b, temperature)
    expected = np.array(expected)
    assert np.array_equal(func, expected)
