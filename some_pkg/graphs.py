import collections


#  Partially Directed Acyclic Graph.
class PDAG:
    def __init__(self, edges=None):
        self.E = set()
        self._Pa = collections.defaultdict(set)
        self._Ch = collections.defaultdict(set)
        if edges is not None:
            self.add_edges(edges)

    def vertices(self):
        return set(self._Pa.keys()) | set(self._Ch.keys())

    def __contains__(self, item):
        return item in self.E

    # Ancestors
    def an(self, x, at=None):
        if at is None:
            at = set()

        for p in self.pa(x):
            if p not in at:
                at.add(p)
                self.an(p, at)

        return at

    # Descendants
    def de(self, x, at=None):
        if at is None:
            at = set()

        for p in self.ch(x):
            if p not in at:
                at.add(p)
                self.de(p, at)

        return at

    # get all oriented edges
    def oriented(self):
        ors = set()
        for x, y in self.E:
            if (y, x) not in self.E:
                ors.add((x, y))
        return ors

    def unoriented(self):
        uors = set()
        for x, y in self.E:
            if (y, x) in self.E:
                uors.add(frozenset({x, y}))
        return uors

    # remove a vertex
    def remove_vertex(self, v):
        for x, y in list(self.E):
            if x == v or y == v:
                self.E.remove((x, y))

        self._Pa.pop(v, None)
        self._Ch.pop(v, None)

        for k, values in self._Pa.items():
            if v in values:
                values.remove(v)
        for k, values in self._Ch.items():
            if v in values:
                values.remove(v)

    def copy(self):
        new_copy = PDAG()
        new_copy.E = set(self.E)
        new_copy._Pa = collections.defaultdict(set)
        new_copy._Ch = collections.defaultdict(set)
        for k, vs in self._Pa.items():
            new_copy._Pa[k] = set(vs)
        for k, vs in self._Ch.items():
            new_copy._Ch[k] = set(vs)

        return new_copy

    # Adjacent
    def is_adj(self, x, y):
        return (x, y) in self.E or (y, x) in self.E

    def add_edges(self, xys):
        for x, y in xys:
            self.add_edge(x, y)

    def add_undirected_edges(self, xys):
        for x, y in xys:
            self.add_undirected_edge(x, y)

    # if y-->x exists, adding x-->y makes x -- y.
    def add_edge(self, x, y):
        assert x != y
        self.E.add((x, y))
        self._Pa[y].add(x)
        self._Ch[x].add(y)

    def add_undirected_edge(self, x, y):
        # will override any existing directed edge
        assert x != y
        self.add_edge(x, y)
        self.add_edge(y, x)

    def orients(self, xys):
        return any([self.orient(x, y) for x, y in xys])

    def orient(self, x, y):
        if (x, y) in self.E:  # already oriented as x -> y?
            if (y, x) in self.E:  # bi-directed?
                self.E.remove((y, x))
                self._Pa[x].remove(y)
                self._Ch[y].remove(x)
                return True
        return False

    def is_oriented_as(self, x, y):
        return (x, y) in self.E and (y, x) not in self.E

    def is_unoriented(self, x, y):
        return (x, y) in self.E and (y, x) in self.E

    def is_oriented(self, x, y):
        return ((x, y) in self.E) ^ ((y, x) in self.E)

    # get neighbors
    def ne(self, x):
        return self._Pa[x] & self._Ch[x]

    # get adjacent vertices
    def adj(self, x):
        return self._Pa[x] | self._Ch[x]

    # get parents
    def pa(self, x):
        return self._Pa[x] - self._Ch[x]

    # get children
    def ch(self, x):
        return self._Ch[x] - self._Pa[x]

    def as_networkx_dag(self):
        assert len(self.unoriented()) == 0
        import networkx as nx
        dg = nx.DiGraph()
        dg.add_edges_from(self.oriented())
        return dg

    def as_networkx_ug(self):
        assert len(self.oriented()) == 0
        import networkx as nx
        ug = nx.Graph()
        ug.add_edges_from(self.unoriented())
        return ug


# x--y--z must be a (shielded or unshielded) non-colider
def meek_rule_3(pdag: PDAG, x, y, z):
    # MR3 x-->w<--z, w--y
    changed = False
    for w in pdag.ch(x) & pdag.ch(z) & pdag.ne(y):
        changed |= pdag.orient(y, w)
    return changed


def meek_rule_2(pdag: PDAG):
    changed = False
    for x, y in list(pdag.E):
        if pdag.is_unoriented(x, y):  # will check y,x, too
            if pdag.ch(x) & pdag.pa(y):  # x-->w-->y
                changed |= pdag.orient(x, y)
    return changed


# x--y--z must be a (shielded or unshielded) non-colider
def meek_rule_4(pdag: PDAG, x, y, z):
    # MR4 z-->w-->x # y-->x
    if pdag.ch(z) & pdag.pa(x):
        return pdag.orient(y, x)
    elif pdag.ch(x) & pdag.pa(z):  # z<--w<--x, z<--y
        return pdag.orient(y, z)
    return False


# x--y--z must be a (shielded or unshielded) non-colider
def meek_rule_1(pdag: PDAG, x, y, z):
    if pdag.is_oriented_as(x, y):
        return pdag.orient(y, z)
    elif pdag.is_oriented_as(z, y):
        return pdag.orient(y, x)
    return False
