import math
from itertools import combinations

from joblib import Parallel, delayed
# kernel for
import numpy as np


def decomposition(xs: list, max_size=None) -> list:
    if max_size is None:
        max_size = len(xs)
    return [comb for i in range(1, max_size + 1) for comb in combinations(xs, i)]


def identity_kernel_func(x, y):
    return 1.0 if x == y else 0.0


# TODO performance
def r_convolution_kernel_func(xs, ys, kernel=identity_kernel_func):
    minlen = min(len(xs), len(ys))
    decomp_x = decomposition(xs, minlen)
    decomp_y = decomposition(ys, minlen)
    k = 0.0
    for x in decomp_x:
        for y in decomp_y:
            k += kernel(x, y)
    return k


def kernel_matrix(kernel_func, X, Y=None):
    """
    create a kernel matrix (numpy ndarray)
    :param kernel_func: a kernel function
    :param X: array-like
    :param Y: array-like
    :return: a kernel matrix
    """
    if Y is None:
        Y = X
    matrix = np.array([len(X), len(Y)])
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            matrix[i, j] = kernel_func(x, y)
    return matrix


def multiset_rbf_kernel(X, Y, gamma):
    """
    (numpy ndarray)
    :param X:
    :param Y:
    :param gamma: gamma in exp(-gamma * || X-Y ||)
    :return: (numpy ndarray)
    """
    matrix = np.array([len(X), len(Y)])

    def rbf(x, y, gamma):
        euc = np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
        euc *= -gamma
        return math.exp(euc)

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            matrix[i, j] = r_convolution_kernel_func(x, y, rbf)


# X: array-like
# Y: array-like
def parallel_kernel_matrix(kernel_func, X, Y=None):
    if Y is None:
        Y = X
    kernel_matrix_as_list = Parallel(n_jobs=8)(delayed(kernel_func)(x, y) for x in X for y in Y)
    matrix = np.reshape(np.array(kernel_matrix_as_list), (len(X), len(Y)))
    return matrix
