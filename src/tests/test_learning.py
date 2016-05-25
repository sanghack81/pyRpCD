import unittest

from pyrcds.model import is_valid_rpath
from pyrcds.rcds import enumerate_rdeps, enumerate_rpaths
from tests.testing_utils import company_rcm, company_schema


class TestLearning(unittest.TestCase):
    def test_something(self):
        schema = company_schema()
        rcm = company_rcm()

        rpaths = set(enumerate_rpaths(schema, 4))
        assert len(rpaths) == 43
        for rpath in enumerate_rpaths(schema, 4):
            assert is_valid_rpath(list(rpath))

        assert rcm.directed_dependencies <= set(enumerate_rdeps(schema, rcm.max_hop))
        assert 22 == len(set(enumerate_rdeps(schema, 4)))
        assert 162 == len(set(enumerate_rdeps(schema, 16)))
        assert 22 == len(set(enumerate_rdeps(schema, 4)))
        assert 162 == len(set(enumerate_rdeps(schema, 16)))

        # skeleton = generate_skeleton(schema)
        # # # draw_skeleton(skeleton)
        # #
        # lg_rcm = linear_gaussians_rcm(rcm)
        # generate_values_for_skeleton(lg_rcm, skeleton)
        #
        # # divide by median?
        # # # TODO kernel not distance!
        # tester = SetKernelRCITester(skeleton, n_job=10)
        # learner = PracticalLearner(schema, rcm.max_hop, skeleton, tester)
        # logger = logging.getLogger(PracticalLearner.__module__)
        # # logger2 = logging.getLogger(PracticalLearner.__module__ + '.' + PracticalLearner.__name__)
        # # logger.setLevel(logging.INFO)
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging.INFO)
        # # logger.addHandler(sh)
        # # print(logger.getEffectiveLevel())
        # # logger.info('working?')
        # # print('above working?')
        # # logger.error('working error?')
        # # logger.warn('working warn?')
        # # logger2.warn('2 working warn?')
        # # logger2.info('2working info?')
        # undirecteds = learner.phase_I()
        # true_undies = {UndirectedRDep(d) for d in rcm.directed_dependencies}
        # recall = len(true_undies & undirecteds) / len(true_undies)
        # prec = len(true_undies & undirecteds) / len(undirecteds)
        # print((prec, recall))


if __name__ == '__main__':
    unittest.main()
