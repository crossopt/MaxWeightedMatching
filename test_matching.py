import unittest
from random import randint
from networkx.generators.random_graphs import erdos_renyi_graph

from matching import match


class MatchingTests(unittest.TestCase):
    @staticmethod
    def matching(gg):
        n = max([max(u, v) for u, v, w in gg])
        G = [[] for i in range(n + 1)]
        for u, v, w in gg:
            G[u].append((v, w))
            G[v].append((u, w))
        mtc = match(G)
        maxw = 0
        used = [False] * (n + 1)
        for u, v, w in gg:
            if mtc[u] == v and mtc[v] == u and used[u] == used[v] == False:
                used[u] = used[v] = True
                maxw += w
        return maxw

    @staticmethod
    def slow_matching(gg):
        maxw = 0
        n = max([max(u, v) for u, v, w in gg])
        for mask in range(2 ** len(gg)):
            used = [False] * (n + 1)
            cntw = 0
            for i in range(len(gg)):
                if mask & (1 << i):
                    u, v, w = gg[i]
                    if used[v] == used[u] == False:
                        used[u] = used[v] = True
                        cntw += w
            maxw = max(maxw, cntw)
        return maxw

    def test_stresses(self):
        for i in range(10000):
            if i % 100 == 0:
                print("i: ", i)
            n = 7
            max_w = 40
            p = 0.5
            g = erdos_renyi_graph(n, p).edges
            gg = []
            for i, j in g:
                gg.append((i, j, randint(0, max_w)))
            ans = MatchingTests.matching(gg)
            exp = MatchingTests.slow_matching(gg)
            self.assertEqual(ans, exp)


if __name__ == '__main__':
    unittest.main()
