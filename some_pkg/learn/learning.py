# Practical Learning Algorithm for RCM
from collections import defaultdict

from some_pkg.relational_domain import RSchema


def enumerate_rdeps(schema: RSchema, h_max: int):
    pass


class RCDLight:
    def __init__(self, schema, ci_tester, h_max):
        self.schema = schema
        self.ci_tester = ci_tester
        self.h_max = h_max
        self.sepsets = defaultdict(lambda: None)

    def phase_I(self):
        candidates = enumerate_rdeps(self.schema, self.h_max)



    def phase_II(self, background_knowledge):
        pass

    def phase_III(self):
        pass


class RpCD:
    def __init__(self, schema, ci_tester, h_max):
        self.schema = schema
        self.ci_tester = ci_tester
        self.h_max = h_max

    def phase_I(self):
        pass

    def phase_II(self):
        pass

    def phase_III(self):
        pass


class PracticalLearner:
    def __init__(self, schema, rdata):
        pass

    def phase_I(self):
        pass

    def phase_II(self):
        pass

    def phase_III(self):
        pass
