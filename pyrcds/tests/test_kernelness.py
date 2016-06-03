import itertools

import numpy as np
from numpy.linalg import eigh
from numpy.random import poisson, randn

from pyrcds.spaces import set_distance_matrix, median_except_diag, eq_size_max_matching_distance, triangle_fixing, \
    eq_size_hausdorff_distance, hausdorff_distance, eq_size_min_perm_distance, max_min_perm_distance, denoise, \
    shift, flip, geomean


def random_set(avoid_empty=False):
    x = poisson(2.)
    if avoid_empty and x == 0:
        x = 1
    return randn(x)


def is_metric(D):
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


if __name__ == '__main__':
    np.random.seed(0)
    dfuncs = [
        eq_size_max_matching_distance,
        max_min_perm_distance,
        eq_size_min_perm_distance,
        hausdorff_distance,
        eq_size_hausdorff_distance
    ]
    n = 20
    it = 0
    epsilon = 1.0e-12
    for _ in range(100):
        sets = [random_set() for _ in range(n)]
        for dfunc in dfuncs:
            D = set_distance_matrix(sets, dfunc)
            fixed_D = triangle_fixing(D, epsilon)
            # if np.any(D != fixed_D) > 0.0:
            #     print('fixed: {}'.format(dfunc.__name__))

            before_fix = is_metric(D)
            after_fixed = is_metric(fixed_D)
            if before_fix < -epsilon:
                print('{:.5f}, {:.5f}'.format(after_fixed, before_fix))
            assert after_fixed >= -epsilon
            K = D2K(fixed_D)
            if min_eigen_value(K) < -1.e-12:
                print(dfunc.__name__)
                print('min eigenvalue: {}'.format(min_eigen_value(K)))
                K1 = denoise(K)
                K2 = flip(K)
                K3 = shift(K)
                # K4 = diffusion(K)
                for k in (K1, K2, K3):
                    me = min_eigen_value(k)
                    print(me)
                    assert -1.0e-12 <= me
                new_K = geomean(K1, K2, K3)
                print(min_eigen_value(new_K))
