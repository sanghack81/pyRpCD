import itertools
import unittest

import numpy as np
from numpy.random import poisson
from numpy.random import randn

from _spaces import hausdorff_distance, eq_size_hausdorff_distance, set_distance_matrix, \
    list_set_distances, max_min_perm_distance, eq_size_min_perm_distance, eq_size_max_matching_distance, \
    flip, triangle_fixing, denoise, shift, list_psd_converters, min_eigen_value, eq_size_max_matching_max_distance, \
    eq_size_max_matching_min_distance
from utils import median_except_diag


def random_set(avoid_empty=False, size=None):
    if size:
        return np.array([random_set() for _ in range(size)])
    else:
        x = poisson(2.)
        if avoid_empty and x == 0:
            x = 1
        return randn(x)


def is_metric(D, tol=-1.0e-12, skip_inf=True):
    """Check reflexivity and symmetry. Returns a measure for violation of triangle inequiality."""
    n = len(D)
    for i in range(n):
        if D[i, i] != 0:
            return False

    for i, j in itertools.combinations(range(n), 2):
        if D[i, j] != D[j, i]:
            return False

    for i, j, k in itertools.combinations(range(n), 3):
        summed = np.sum([D[i, j], D[j, k], D[i, k]])
        maxed = np.max([D[i, j], D[j, k], D[i, k]])

        if summed == float('inf') and maxed == float('inf'):
            if skip_inf:
                continue
            else:
                return False

        v = summed - 2 * maxed
        if v < tol:
            return False

    return True


def rbf_D2K(D):
    mm = median_except_diag(D)
    if mm != 0:
        return np.exp(-(D * D) / mm)
    else:
        return np.exp(-(D * D))


class TestSpaces(unittest.TestCase):
    def test_metric(self):
        candidates = list_set_distances()
        non_metrics = list()
        tries = 50
        for _ in range(tries):
            for func in list(candidates):
                n = 20
                sets = random_set(True, n)
                D = set_distance_matrix(sets, set_metric=func)
                if not is_metric(D):
                    non_metrics.append(func)
                    candidates.remove(func)
        for nm in non_metrics:
            print('non_metric: {}'.format(nm.__name__))
        for m in candidates:
            print('seems metric: {}'.format(m.__name__))
        assert hausdorff_distance in candidates
        assert eq_size_hausdorff_distance in candidates
        assert eq_size_min_perm_distance in candidates
        assert max_min_perm_distance in candidates
        assert eq_size_max_matching_distance in non_metrics
        assert eq_size_max_matching_min_distance in non_metrics
        assert eq_size_max_matching_max_distance in non_metrics

    def test_triangle_fixing(self):
        pass

    def test_fix_kernels(self):
        tries = 2000
        n = 20
        for _ in range(tries):
            sets = random_set(np.random.rand() < 0.5, n)
            print('', flush=True)
            for conv in list_psd_converters():
                for setd in list_set_distances():
                    D = set_distance_matrix(sets, set_metric=setd)
                    K = rbf_D2K(D)
                    if min_eigen_value(K) < 0:
                        assert -1.0e-13 <= min_eigen_value(conv(K)), \
                            'failed with {} on D with {}'.format(conv.__name__, setd.__name__)
                        print(':', end='', flush=True)
                    else:
                        print('.', end='', flush=True)

    def test_observe_D2K2D_stability(self):
        tries = 2000
        for _ in range(tries):
            n = 20
            sets = random_set(True, n)
            D = set_distance_matrix(sets, set_metric=hausdorff_distance)
            D = triangle_fixing(D)
            K = rbf_D2K(D)
            Kd = denoise(K)
            Kf = flip(K)
            Ks = shift(K)
            gapd = np.max(np.abs(K - Kd))
            gapf = np.max(np.abs(K - Kf))
            gaps = np.max(np.abs(K - Ks))
            if not (gapd <= gapf and gapd <= gaps):
                print(gapd, gapf, gaps)

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

    def test_eq_size_max_matching_distance(self):
        x = np.array([0.1, 0.3, 0.5, 0.7], dtype=float)
        y = x + 0.02
        y[0] = y[0] + 0.01
        for _ in range(10):
            np.random.shuffle(y)
            assert np.allclose(eq_size_max_matching_distance(x, y), 0.09)
            assert np.allclose(eq_size_max_matching_min_distance(x, y), 0.02)
            assert np.allclose(eq_size_max_matching_max_distance(x, y), 0.03)

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
