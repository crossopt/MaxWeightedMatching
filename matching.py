from itertools import chain
from collections import deque

INF = 1e9 + 8


class Blossom:
    """ Class containing the inner structure of a compressed blossom. """

    def __init__(self, v):
        self.base = v  # the base vertex
        self.children = []  # a list of subblossoms in the cycle order
        self.parent_edge = None  # edge from outside the blossom that gave it the current label
        self.edges = []  # list of edges on the blossom in the cycle order
        self.parent = None  # link to parent blossom that contains the current blossom
        self.z = 0  # blossom's dual variable
        self.label = 1  # 0 for free vertices, 1 for S-vertices, 2 for T-vertices

    def add(self, other_blossom, edge):
        # Adds another subblossom to the blossom through the given edge.
        self.children.append(other_blossom)
        self.edges.append(edge)
        other_blossom.parent = self
        other_blossom.parent_edge = edge

    def family(self):
        # Returns a list of all the vertices contained in the blossom.
        return [self.base] if not self.children else list(chain(*[kid.family() for kid in self.children]))

    def locate(self, v):
        # Returns the index in the subblossom list of the subblossom containing v.
        for index in range(len(self.children)):
            if v in self.children[index].family():
                return index

    def fix(self):
        # Rotates the blossom's list of subblossoms and edges to start from the base vertex.
        index = self.locate(self.base)
        self.children = self.children[index:] + self.children[:index]
        self.edges = self.edges[index:] + self.edges[:index]

    def clear_labels(self):
        # Clears all labels on the blossom and its subblossoms.
        self.label = 0
        for child in self.children:
            child.clear_labels()


def match(G):
    """
    Implementation of the http://www.cs.kent.edu/~dragan/GraphAn/p23-galil.pdf algorithm.
    Returns the maximum weighted matching for graph G: a list of matches or None for vertices.
    The format of G: lists of (u, w) for each v, where the edge u -> v weighs w.
    Time complexity is Poly(n).
    """
    if sum([len(i) for i in G]) == 0:
        return [None] * len(G)

    n = len(G)
    mate = [None] * n  # The final matching.
    blossoms = [Blossom(v) for v in range(n)]  # Stores the topmost blossom with base v.
    base_blossoms = [blossoms[v] for v in range(n)]  # Stores trivial blossoms for each vertex.

    us = [max(w for _, w in chain(*G))] * n  # Stores the dual variables for vertices.
    q = deque()  # Queue of vertices to update from.
    bfs_used, blossom_used = set(), set()

    get_top = lambda bv: get_top(bv.parent) if bv.parent else bv  # topmost blossom containing bv
    get_blossom = lambda v: get_top(base_blossoms[v])  # topmost blossom containing vertex v
    get_blossoms = lambda: set(map(get_blossom, range(n)))  # set of topmost blossoms
    move_back = lambda bv: get_blossom(bv.parent_edge[0])  # blossom from which bv got its edge
    get_base = lambda v: get_blossom(v).base  # the base of the blossom containing v
    get_label = lambda v: get_blossom(v).label  # the label of the blossom containing v

    def add_vertices(pred):
        """ Add all unused vertices that satisfy a predicate to the bfs queue. """
        for v in range(n):
            if pred(v) and v not in bfs_used:
                bfs_used.add(v)
                q.append(v)

    def set_r1(v, edge):
        """ Apply r1: set a blossom's label to S, add its vertices to the queue. """
        bv = get_blossom(v)
        bv.parent_edge, bv.label = edge, 1
        add_vertices(lambda v: get_blossom(v) == bv)

    def set_r2(v, edge):
        """ Apply r2: set a blossom's label to T and its mate's to S. """
        bv = get_blossom(v)
        bv.parent_edge, bv.label = edge, 2
        base = get_base(v)  # only the base can have a mate in another blossom
        set_r1(mate[base], (base, mate[base]))

    def get_lca(u, v):
        """ Returns the base vertex of the lca of blossoms u and v, None if it does not exist. """
        lca_used, lca_vsed = set(), set()
        while u not in lca_used:
            lca_used.add(u)
            if u.parent_edge:
                u = move_back(u)
        while v not in lca_vsed:
            lca_vsed.add(v)
            if v in lca_used:
                return v.base
            for step in range(2):  # step to mate, then to S-vertex
                if not v.parent_edge:
                    return None
                v = move_back(v)
        return None

    def mark_blossom(lca, bv):
        """ Return list of blossoms on path to lca from bv to add to lca's blossom. """
        mark_blossoms, mark_edges = [], []
        while bv.base != lca:
            if bv not in blossom_used:
                blossom_used.add(bv)
                mark_blossoms.append(bv)
                mark_edges.append(bv.parent_edge)
            if not bv.parent_edge:
                break
            bv = move_back(bv)
        return mark_blossoms, mark_edges

    def contract_blossom(lca, u, v):
        """ Create a new blossom with base in lca containing u and v. """
        # Create a new blossom for lca in place of the existing one
        blossom_used.clear()
        old_blossom = blossoms[lca]
        blossoms[lca] = Blossom(lca)
        blossom_used.add(blossoms[lca])
        blossoms[lca].parent_edge = old_blossom.parent_edge

        # Add subblossoms to lca's blossom in cycle order
        mark_blossoms, mark_edges = mark_blossom(lca, get_blossom(u))
        mark_blossoms = [old_blossom] + mark_blossoms[::-1]
        mark_edges = [i[::-1] for i in mark_edges][::-1] + [(v, u)]
        new_marks = mark_blossom(lca, get_blossom(v))
        for bv, ed in zip(mark_blossoms + new_marks[0], mark_edges + new_marks[1]):
            blossoms[lca].add(bv, ed)
        # Add updated vertices to bfs
        add_vertices(lambda v: get_blossom(v) in blossom_used and get_label(v) != 0)

    def find_path():
        """ The main bfs. Returns True if a new augmenting path was found. """
        bfs_used.clear()
        q.clear()
        add_vertices(lambda v: get_label(v) == 1)

        while len(q) != 0:
            v = q.popleft()
            for u, w in G[v]:
                if us[v] + us[u] - 2 * w > 0 or \
                        mate[get_base(v)] == get_base(u) or get_blossom(v) == get_blossom(u):
                    continue  # edge is unusable, in matching and blossom-internal respectively
                if get_label(u) == 0:  # free vertex
                    set_r2(u, (v, u))
                elif get_label(u) == 1:  # S-vertex
                    lca = get_lca(get_blossom(v), get_blossom(u))
                    if lca is not None:  # a new blossom was found
                        contract_blossom(lca, v, u)
                    else:  # found an augmenting path
                        recover_path(v, u)
                        recover_path(u, v)
                        return True
        return False

    def t_expand(bv):
        """ Expands a T-blossom and fixes its subblossoms' labels. """
        if not bv.children:
            return

        # Cycle direction is dependent on the parity of the base's index
        init = bv.locate(get_base(bv.parent_edge[1]))
        if init % 2:
            edges = [bv.edges[0]] + bv.edges[init:] + [bv.edges[0]]
            edges = [i[::-1] for i in edges]  # edges point in the wrong direction
            free_children = bv.children[:init]
        else:
            free_children = bv.children[len(bv.children):init:-1]
            edges = bv.edges[init::-1]

        last_edge = bv.parent_edge
        t_len = (len(bv.children) - init) if init % 2 else init
        # Set new labels and edges along cycle
        for i in range(1, t_len, 2):
            x, y = last_edge
            last_edge = edges[i + 1]
            set_r2(y, (x, y))
        # Set the base as well, not modifying its edge
        bv.children[0].parent_edge, bv.children[0].label = last_edge, 2

        # The rest of the cycle becomes free
        for child in free_children[1:]:
            child.label = 0

    def expand_blossom(bv, recursive_expand):
        """ Expands the blossom bv and possibly its subblossoms. """
        for kid in bv.children:
            kid.parent = None
            blossoms[kid.base] = kid
            if recursive_expand and not kid.z:
                expand_blossom(kid, True)
        if bv.label == 2 and not recursive_expand:
            t_expand(bv)

    def recover_blossom(bv, v):
        """ Fixes a blossom's matching to free the vertex v and set it to base. """
        if not bv.children:
            return

        # Get the proper layer to augment
        upper = base_blossoms[v]
        while upper.parent and upper.parent != bv:
            upper = upper.parent
        if upper.parent == bv:
            recover_blossom(upper, v)

        # Order of blossom recovery is dependent on the parity of its index in order
        init = bv.locate(v)
        if init % 2:
            children = bv.children[init:] + [bv.children[0]]
            edges = bv.edges[init:] + [bv.edges[0]]
        else:
            children = bv.children[:init + 1][::-1]
            edges = bv.edges[:init][::-1]
            edges = [i[::-1] for i in edges]  # edges point in the wrong direction

        for index in range(1, len(children) - 1, 2):
            x, y = edges[index]
            recover_blossom(children[index], y)
            recover_blossom(children[index + 1], x)
            mate[x], mate[y] = y, x
        bv.base = v
        bv.fix()
        blossoms[v] = bv

    def recover_path(v, u):
        """ Augments the edge path from v towards its mate u. """
        bv = get_blossom(v)
        recover_blossom(bv, v)
        mate[v], mate[u] = u, v

        while bv.parent_edge:
            bu = move_back(bv)
            v, u = bu.parent_edge
            recover_blossom(bu, u)
            mate[u] = v
            bv = get_blossom(v)
            recover_blossom(bv, v)
            mate[v] = u

    # Main algorithm cycle
    for _ in range(n):
        # Clear labels, set label to S for free blossoms.
        for v in range(n):
            get_blossom(v).clear_labels()
        for v in range(n):
            if mate[get_base(v)] is None:
                set_r1(v, None)

        # Erase parent edges for upper levels.
        for v in range(n):
            get_blossom(v).parent_edge = None

        while not find_path():
            d1 = INF  # Vertices' dual variables.
            for v in range(n):
                d1 = min(d1, us[v])

            d2 = INF  # Edge between S-vertex and free vertex.
            for v in filter(lambda v: get_label(v) == 1, range(n)):
                for u, w in filter(lambda u: get_label(u[0]) == 0, G[v]):
                    d2 = min(d2, us[v] + us[u] - 2 * w)

            d3 = 2 * INF  # Edge between two S-vertices in different blossoms.
            for v in filter(lambda v: get_label(v) == 1, range(n)):
                for u, w in filter(lambda u: get_label(u[0]) == 1 and get_blossom(u[0]) != get_blossom(v), G[v]):
                    d3 = min(d3, us[v] + us[u] - 2 * w)
            d3 //= 2

            d4 = INF  # Non-trivial T-blossoms.
            for bv in filter(lambda bv: bv.children and bv.label == 2, get_blossoms()):
                d4 = min(d4, bv.z)

            delta = min(d1, d2, d3, d4)

            # Update values
            for v in range(n):
                if get_label(v) == 1:  # S-vertex
                    us[v] -= delta
                elif get_label(v) == 2:  # T-vertex
                    us[v] += delta
            for bv in get_blossoms():
                if bv.children and bv.label == 1:  # S-blossom
                    bv.z += delta
                elif bv.children and bv.label == 2:  # T-blossom
                    bv.z -= delta

            if delta == d1:  # Optimal solution is found.
                return mate
            elif delta == d4:  # Zero z-value T-blossoms expansion.
                for bv in filter(lambda bv: not bv.z and bv.label == 2, get_blossoms()):
                    expand_blossom(bv, False)

        # Recursively expand all blossoms at end of stage.
        for bv in filter(lambda bv: not bv.z, get_blossoms()):
            expand_blossom(bv, True)
    return mate


if __name__ == '__main__':
    # Read input (vertices from 0 to n - 1)
    n, m = map(int, input().split())
    G = [[] for i in range(n)]
    vals = [dict() for i in range(n)]
    for i in range(m):
        u, v, w = list(map(int, input().split()))
        vals[u][v] = vals[v][u] = w
        G[u].append((v, w))
        G[v].append((u, w))
    result = match(G)
    sm = sum([vals[v][result[v]] for v in range(n) if result[v] is not None])
    print(sm // 2)
    print(*result)
