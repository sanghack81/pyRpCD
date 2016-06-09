# Relational Conditional Independence
# Relational, Kernel Conditional Independence Permutation Test
import typing

from numpy import exp, array, sqrt
from numpy.random import choice
from sklearn.metrics.pairwise import pairwise_distances

from pygk.utils import as_column
from pykcipt.kcipt import KCIPT, KCIPTResult, KIPT
from pyrcds._spaces import set_distance_matrix, weisfeiler_lehman_vertex_kernel, eq_size_max_matching_distance, \
    euclidean, distance_matrix
from pyrcds.domain import RSkeleton, ImmutableRSkeleton
from pyrcds.model import RVar, flatten
from pyrcds.utils import median_except_diag


def rbf_D2K(D):
    mm = median_except_diag(D)
    if mm != 0:
        return exp(-(D * D) / mm)
    else:
        return exp(-(D * D))


def sums(*args):
    assert args
    assert len(args) > 0
    temp = 0
    for arg in args:
        temp += arg
    return temp


def multiplys(*args):
    assert args
    assert len(args) > 0
    temp = 1
    for arg in args:
        temp *= arg
    return temp


def dump_values(attr, skeleton):
    return array([item[attr] for item in skeleton.items(skeleton.schema.item_class_of(attr))])


class CITester:
    def __init__(self):
        pass

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> KCIPTResult:
        raise NotImplementedError()

    @property
    def is_p_value_available(self):
        raise NotImplementedError()


class SetKernelRCITester(CITester):
    """Set kernel (multi-instance kernel) based tester"""

    def __init__(self, skeleton: RSkeleton, n_jobs=-1, maxed=None, **options):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.n_jobs = n_jobs
        self.options = options
        self.maxed = maxed

        self.median_dist = dict()
        for attr in skeleton.schema.attrs:
            dist = pairwise_distances(as_column(dump_values(attr, skeleton)))
            self.median_dist[attr] = median_except_diag(dist)

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()) -> KCIPTResult:
        assert x != y
        assert x not in zs and y not in zs

        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=True)
        if self.maxed is not None and len(data) > self.maxed:
            data = data[choice(len(data), self.maxed, replace=False), :]

        K = [None] * (2 + len(zs))
        Ds = [None] * (2 + len(zs))
        for i, rvar in enumerate((x, y, *zs)):
            Ds[i] = set_distance_matrix(data[:, i])
            K[i] = rbf_D2K(Ds[i])

        if zs:
            return KCIPT(multiplys(K[0], *K[2:]), K[1],
                         sqrt(sums(*[D ** 2 for D in Ds[2:]])),
                         n_jobs=self.n_jobs, **self.options)
        else:
            return KIPT(K[0], K[1], n_jobs=self.n_jobs, **self.options)

    @property
    def is_p_value_available(self):
        return True


class GraphKernelRCITester(CITester):
    def __init__(self, skeleton: RSkeleton,
                 vertex_kernel=weisfeiler_lehman_vertex_kernel,
                 attr_metrics=euclidean,
                 alpha=0.05, n_jobs=-1, maxed=None,
                 **options):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.options = options

        self.VK, ordered_items = vertex_kernel(self.skeleton, **options)
        self.index_of = {item: index for index, item in enumerate(ordered_items)}
        self.maxed = maxed

        attrs = self.skeleton.schema.attrs

        if attr_metrics is None:
            attr_metrics = euclidean

        if not isinstance(attr_metrics, dict) and callable(attr_metrics):
            attr_metrics = {attr: attr_metrics
                            for attr in attrs}

        ITEM, VALUE = 0, 1

        def sim_func(iv1, iv2):
            return self.VK[self.index_of[iv1[ITEM]], self.index_of[iv2[ITEM]]]

        # squared maximum matching distance
        def smd(ivs1, ivs2, dist_iv_pair):
            return eq_size_max_matching_distance(ivs1, ivs2, dist_iv_pair, sim_func)

        self.set_metrics = {a: (lambda ivs1, ivs2: smd(ivs1, ivs2,
                                                       lambda iv1, iv2: attr_metrics[a](iv1[VALUE], iv2[VALUE])))
                            for a in attrs}

    def ci_test(self, x: RVar, y: RVar, zs: typing.Set[RVar] = frozenset()):
        assert x != y
        assert x not in zs and y not in zs
        assert y.is_canonical or x.is_canonical
        if x.is_canonical:
            x, y = y, x

        # data is not made of values, it is made of (item, value) pairs.
        data = flatten(self.skeleton, (x, y, *zs), with_base_items=False, value_only=False, n_jobs=self.n_jobs)
        if self.maxed is not None and len(data) > self.maxed:
            data = data[choice(len(data), self.maxed, replace=False), :]

        K = [None] * (2 + len(zs))
        Ds = [None] * (2 + len(zs))
        for i, rvar in enumerate((x, y, *zs)):
            Ds[i] = distance_matrix(data[:, i], metric=self.set_metrics[rvar.attr])
            K[i] = rbf_D2K(Ds[i])

        if zs:
            return KCIPT(multiplys(K[0], *K[2:]), K[1],
                         sqrt(sums(*[D ** 2 for D in Ds[2:]])),
                         n_jobs=self.n_jobs, **self.options)
        else:
            return KIPT(K[0],
                        K[1],
                        n_jobs=self.n_jobs,
                        **self.options)

    @property
    def is_p_value_available(self):
        return True
