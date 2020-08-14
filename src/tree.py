from collections.abc import MutableMapping, Set, Iterable

import networkx as nx


class Tree(nx.DiGraph):
    def parent(self, n):
        parents = list(self._pred[n])
        if len(parents) > 1:
            raise ValueError(f'node {n!r} has more than one parent')
        return parents[0] if parents else None

    def path(self, source, target):
        "Find path to *node* from *root*."
        u = target
        path = [u]
        while u != source:
            u = self.parent(u)
            if u is None:
                raise ValueError(f'no path from {source!r} to {target!r}')
            path.append(u)
        return path[::-1]

    @property
    def root(self):
        return self.graph['root']

    @root.setter
    def root(self, root):
        self.graph['root'] = root

    def check(self):
        check_digraph_consistency(self)


class EdgeDataMap(MutableMapping, Set, Iterable):
    __slots__ = ['_adj', '_key']

    def __getstate__(self):
        return {'_adj': self._adj, '_key': self._key}

    def __setstate__(self, state):
        self._adj = state['_adj']
        self._key = state['_key']

    def __init__(self, graph, key):
        self._adj = graph._adj
        self._key = key

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def __getitem__(self, n):
        u, v = n
        return self._adj[u][v][self._key]

    def __setitem__(self, n, val):
        u, v = n
        self._adj[u][v][self._key] = val

    def __delitem__(self, n):
        u, v = n
        del self._adj[u][v][self._key]

    def __contains__(self, n):
        u, v = n
        return ((u in self._adj) and 
                (v in self._adj[u]) and 
                (self._key in self._adj[u][v]))

    def __str__(self):
        return str(dict(iter(self)))

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self._adj, self._key)

    def values(self):
        key = self._key
        return (dd_uv[key]
                for nbrs in self._adj.values()
                for dd_uv in nbrs.values()
                if key in dd_uv)

    def items(self):
        key = self._key
        return (((u, v), dd_uv[key])
                for u, nbrs in self._adj.items()
                for v, dd_uv in nbrs.items()
                if key in dd_uv)


class NodeDataMap(MutableMapping, Set, Iterable):
    __slots__ = ['_node', '_key']

    def __getstate__(self):
        return {'_node': self._node, '_key': self._key}

    def __setstate__(self, state):
        self._node = state['_node']
        self._key  = state['_key']

    def __init__(self, graph, key):
        self._node = graph._node
        self._key = key

    def __len__(self):
        return len(self._node)

    def __iter__(self):
        return iter(self._node)

    def __getitem__(self, u):
        return self._node[u][self._key]

    def __setitem__(self, u, val):
        self._node[u][self._key] = val

    def __delitem__(self, u):
        del self._node[u][self._key]

    def __contains__(self, u):
        return ((u in self._node) and 
                (self._key in self._node[u]))

    def __str__(self):
        return str(dict(iter(self)))

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self._node, self._key)

    def values(self):
        key = self._key
        return (dd[key] for dd in self._node.values() if key in dd)

    def items(self):
        key = self._key
        return ((u, dd[key]) for u, dd in self._node.items() if key in dd)


def max_path(T, u, *, key):
    "Viterbi-like maximal path"
    while True:
        yield u
        nbrs = T.adj[u]
        if not nbrs:
            break
        u = max(nbrs, key=key)


def subtree(T, new_root):
    S = type(T)()
    node_map = {}
    m = lambda u: node_map.setdefault(u, len(node_map))
    S.graph = T.graph.copy()
    S.root = m(new_root)
    stack = [new_root]
    while stack:
        u = stack.pop()
        mu = m(u)
        S._node[mu] = T._node[u].copy()
        S._succ[mu] = {m(v): dd.copy() for v, dd in T._succ[u].items()}
        for mv, dd in S._succ[mu].items():
            S._pred[mv] = {mu: dd}
        stack.extend(T._succ[u])
    return S


def assert_eq_diff(a,b):
    assert a == b, f'a==b, a-b: {a-b!r}, b-a: {b-a!r}'


def check_digraph_consistency(G):
    assert_eq_diff(set(G._succ), set(G._node))
    assert_eq_diff(set(G._succ), set(G._pred))
    assert_eq_diff(set(), set(v for u in G for v in G._succ[u]) - set(G._node))
    assert_eq_diff(set(), set(u for v in G for u in G._pred[v]) - set(G._node))
    # Make sure all successor edges also exist as predecessor edges
    assert_eq_diff(set((u, v) for u in G for v in G._succ[u]),
                   set((u, v) for v in G for u in G._pred[v]))
