import itertools
import unittest

import numpy as np
from numpy.linalg import eigh
from numpy.random import poisson
from numpy.random import randn

from spaces import hausdorff_distance, eq_size_hausdorff_distance, median_except_diag


def random_set(avoid_empty=False):
    x = poisson(2.)
    if avoid_empty and x == 0:
        x = 1
    return randn(x)


def is_metric(D, n):
    """Check reflexivity and symmetry. Returns a measure for violation of triangle inequiality."""
    for i in range(n):
        assert D[i, i] == 0

    for i, j in itertools.combinations(range(n), 2):
        assert D[i, j] == D[j, i]

    max_violation = 0
    for i, j, k in itertools.combinations(range(n), 3):
        summed = np.sum([D[i, j], D[j, k], D[i, k]])
        maxed = np.max([D[i, j], D[j, k], D[i, k]])
        if summed == float('inf') and maxed == float('inf'):
            continue

        v = summed - 2 * maxed
        if v < 0 and v < max_violation:
            max_violation = v

    return max_violation


def min_eigen_value(K):
    w, _ = eigh(K)
    return min(w)


def D2K(D):
    mm = median_except_diag(D)
    if mm == float('inf'):
        mm = 1
    return np.exp(-D / mm)


class TestSpaces(unittest.TestCase):
    def test_triangle_fixing(self):
        pass

    def test_fix_kernels(self):
        # denoise
        # shift
        # diffusion
        # flip
        pass

    def test_geomean(self):
        pass

    def test_normalize_by_diag(self):
        pass

    def test_hausdorff_distance(self):
        xs = [1, 2, 3]
        ys = [2, 3, 4]
        assert hausdorff_distance(xs, xs) == 0
        assert hausdorff_distance(ys, ys) == 0
        assert hausdorff_distance(xs, ys) == 1
        assert hausdorff_distance(ys, xs) == 1
        xs = [1, 2, 3]
        ys = [2, 3]
        assert hausdorff_distance(xs, ys) == 1
        assert hausdorff_distance(ys, xs) == 1
        xs = [1, 2, 3]
        ys = [2, 4]
        assert hausdorff_distance(xs, ys) == 1
        assert hausdorff_distance(ys, xs) == 1
        xs = [1, 2, 3]
        ys = [2, 5]
        assert hausdorff_distance(xs, ys) == 2
        assert hausdorff_distance(ys, xs) == 2
        xs = [1, 2, 3]
        ys = [1, 1.5, 2, 2.5, 3]
        assert hausdorff_distance(xs, ys) == 0.5
        assert hausdorff_distance(ys, xs) == 0.5
        xs = [1, 2, 3]
        ys = []
        assert hausdorff_distance(xs, ys) == float('inf')
        assert hausdorff_distance(ys, xs) == float('inf')
        xs = []
        ys = []
        assert hausdorff_distance(xs, ys) == 0
        assert hausdorff_distance(ys, xs) == 0

    def test_max_match(self):
        pass

    def test_eq_size_max_matching_distance(self):
        pass

    def test_eq_size_hausdorff_distance(self):
        xs = [1, 2, 3]
        ys = [2, 3, 4]
        assert eq_size_hausdorff_distance(xs, xs) == 0
        assert eq_size_hausdorff_distance(ys, ys) == 0
        assert eq_size_hausdorff_distance(xs, ys) == 1
        assert eq_size_hausdorff_distance(ys, xs) == 1
        xs = [1, 2, 3]
        ys = [2, 3]
        assert eq_size_hausdorff_distance(xs, ys) == float('inf')
        assert eq_size_hausdorff_distance(ys, xs) == float('inf')
        xs = [1, 2, 3]
        ys = [2, 4]
        assert eq_size_hausdorff_distance(xs, ys) == float('inf')
        assert eq_size_hausdorff_distance(ys, xs) == float('inf')
        xs = [1, 2, 3]
        ys = [2, 5]
        assert eq_size_hausdorff_distance(xs, ys) == float('inf')
        assert eq_size_hausdorff_distance(ys, xs) == float('inf')
        xs = [1, 2, 3]
        ys = [1, 1.5, 2]
        assert eq_size_hausdorff_distance(xs, ys) == 1
        assert eq_size_hausdorff_distance(ys, xs) == 1
        xs = [1, 2, 3]
        ys = []
        assert eq_size_hausdorff_distance(xs, ys) == float('inf')
        assert eq_size_hausdorff_distance(ys, xs) == float('inf')
        xs = []
        ys = []
        assert eq_size_hausdorff_distance(xs, ys) == 0
        assert eq_size_hausdorff_distance(ys, xs) == 0

    def test_eq_size_min_perm_distance(self):
        pass

    def test_max_min_perm_distance(self):
        pass

    def set_distance_matrix(self):
        pass

    def test_weisfeiler_lehman_vertex_kernel(self):
        pass

    def test_median_except_diag(self):
        D = np.array([[1, 3], [2, 5]])
        assert median_except_diag(D) == 2.5
        D = np.array([[1, 3, 4], [2, 4, 5], [4, 6, 6]])
        # 3,4,2,5,4,6
        assert median_except_diag(D) == 4
        D = np.array([[1, 3, 4], [2, 4, 5], [4, 100000, 6]])
        # 3,4,2,5,4,100000
        assert median_except_diag(D) == 4
        D = np.array([[1, 3, 4], [2, 4, 5], [5, 100000, 6]])
        # 3,4,2,5,5,100000
        assert median_except_diag(D) == 4.5


if __name__ == '__main__':
    unittest.main()
