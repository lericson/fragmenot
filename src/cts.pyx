#cython: language_level=3
"Plan a path for exploration in a state graph"

import sys
import logging

import numpy as np
import networkx as nx
from networkx.utils import pairwise
cimport cython
cimport numpy as np

import gui


log = logging.getLogger(__name__)


# Label maker. The advantage of doing this is that we can never have a bug
# where a new node gets assigned an already existing node label. We might run
# out of node labels though, and they may get pretty long.
cdef int _label = 0
cdef inline int next_label() nogil:
    global _label
    _label += 1
    return _label


def statstr(a):
    a = np.atleast_1d(a)
    if len(a) == 0:
        return 'N=0, min=?, mean=?, max=?, std=?'
    elif len(a) == 1:
        return f'N=1, min={a[0]}, mean={a[0]}, max={a[0]}, std=?'
    else:
        return ', '.join([f'N={len(a)}',
                          f'min={np.min(a):.5g}', f'mean={np.mean(a):.5g}',
                          f'max={np.max(a):.5g}', f'std={np.std(a):.5g}'])


cdef class EdgeDataMap():
    cdef dict _adj
    cdef object _key

    def __cinit__(self, graph, key):
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
        return [dd_uv[key]
                for nbrs in self._adj.values()
                for dd_uv in nbrs.values()
                if key in dd_uv]


cdef class NodeDataMap():
    cdef dict _node
    cdef object _key

    def __cinit__(self, graph, key):
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
        return [dd[self._key] for dd in self._node.values() if self._key in dd]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double sum_where_1d(double [::1] a, unsigned char [::1] mask) nogil:
    cdef Py_ssize_t k = a.shape[0]
    cdef double r = 0.0
    for i in range(k):
        if mask[i]:
            r += a[i]
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint recurred(NodeDataMap Score, list path, list path_s, int state, double score):
    cdef int i
    for i in range(len(path) - 1):
        if path_s[i] == state and score <= Score[path[i]]:
            return True
    return False


def expand(object T, *, object roadmap,
           int    max_depth = 50,
           int    steps     = 15000,
           int    max_size  = 30000,
           double lam       = 50e-2,
           int    min_faces = 1):

    cdef object G = roadmap

    cdef EdgeDataMap D          = EdgeDataMap(G, 'd')
    cdef EdgeDataMap Vis_faces  = EdgeDataMap(G, 'vis_faces')

    cdef NodeDataMap S          = NodeDataMap(T, 's')
    cdef NodeDataMap Seen       = NodeDataMap(T, 'seen')
    cdef NodeDataMap Score      = NodeDataMap(T, 'score')
    cdef NodeDataMap Dist       = NodeDataMap(T, 'dist')
    cdef NodeDataMap Depth      = NodeDataMap(T, 'depth')
    cdef NodeDataMap N_child    = NodeDataMap(T, 'n_child')
    cdef NodeDataMap Unvisited  = NodeDataMap(T, 'unvisited')

    cdef int step, j, u, v, w, s_u, s_v
    cdef list unvisited_u, path, path_s

    log.info('computing weights')

    face_vis   = roadmap.graph['vis_faces']
    face_covis = roadmap.graph['covis_faces']
    face_area  = roadmap.graph['area_faces']

    vis_unseen = face_vis & ~Seen[T.root]

    # Compute face covisibility area, excluding seen faces. We do this at a
    # smaller length scale because of integer mathematics. Do not change to
    # floats, it will waste vast amounts of memory.
    face_degree = face_covis.dot((1e5*face_area*vis_unseen).astype(np.uint32))/1e5
    face_degree[~face_vis] = 0

    #for (i,) in zip(*np.nonzero(face_vis)):
    #    assert (face_degree[i] == np.dot(face_covis[i], vis_unseen)), f'{i}: {face_degree[i]}'

    face_score = 1e2*np.exp(-1e-1*(face_degree.clip(0, 1000)))
    face_score[face_degree < min_faces] = 1e-12
    face_score[~vis_unseen]             = 1e-12

    gui.update_face_hsv(hues=0.8/face_score.max()*face_score, layer='score')
    #gui.update_face_hsv(hues=0.8/face_degree.max()*face_degree)

    log.info('  face_area: %s', statstr(face_area[vis_unseen]))
    log.info('face_degree: %s', statstr(face_degree[vis_unseen]))
    log.info(' face_score: %s', statstr(face_score[vis_unseen]))
    log.info('          d: %s', statstr(list(D.values())))

    log.info('expanding tree %d steps (initial size: %d, max size: %d)',
             steps, len(T), max_size)

    cdef int    best_node  = T.graph['best_node']
    cdef double best_score = T.graph['best_score']

    """
    from collections import deque
    stack = deque([([T.root], n) for n in G.adj[S[T.root]]])
    """

    cdef dict Gadj = G._adj
    cdef dict Tadj = T._adj
    cdef object N_child_get = N_child.__getitem__
    cdef np.ndarray vis
    cdef np.ndarray seen_u
    cdef double score_u, score, gain, distfac

    cdef int n_recurred = 0, n_inserted = 0

    for step in range(steps):

        ## SELECTION: find an unexplored branching point in the tree. Result is
        ## the chosen next state S[v], and its path.

        """
        path, s_v = stack.popleft()
        u = path[-1]
        s_u = S[u]

        # Check if this edge has been visited before in this direction along
        # this path. If so, skip it.
        score_u = Score[u]
        s_uv = s_u, s_v
        if any(score_u <= Score[w_] for (w, w_) in pairwise(path) if s_uv == (S[w], S[w_])):
            continue

        v = next_label()
        T.add_node(v, s=s_v)
        T.add_edge(u, v)
        path = path + [v]
        stack.extend([(path, v_) for v_ in Gadj[s_v]])
        """

        path, u = [], T.root
        path_s = []

        # This loop generates a path = [..., u], and a successor state s_v of
        # s_u according to G.
        for j in range(max_depth):

            s_u = S[u]
            path.append(u)
            path_s.append(s_u)

            if u not in Unvisited:
                unvisited_u = Unvisited[u] = list(Gadj[s_u])
            else:
                unvisited_u = Unvisited[u]

            if unvisited_u:
                # The tree node has unvisited neighboring states, let s_v be
                # one of them.
                s_v = unvisited_u.pop()
                break

            nbrs = list(Tadj[u])
            if not nbrs:
                #raise RuntimeError('node is leaf and no neighbor states')
                v = u
                u = path[-1]
                s_v = S[v]
                s_u = S[u]
                # Set a really high N_child so it doesn't get visited again
                N_child[u] += 100000
                break
            else:
                u = min(nbrs, key=N_child_get)
        else:
            raise RuntimeError('ran out of attempts')

        score_u = Score[u]

        #if any(score_u <= Score[w_] for (w, w_) in pairwise(path) if s_uv == (S[w], S[w_])):
        if recurred(Score, path, path_s, s_u, score_u):
             n_recurred += 1
             continue
        n_inserted += 1

        # EXPANSION: add new node to search tree
        v = next_label()
        T.add_node(v, s=s_v)
        T.add_edge(u, v)

        seen_u   = Seen[u]
        Dist[v]  = <double>Dist[u] + <double>D[s_u, s_v]
        Depth[v] = <int>Depth[u] + 1
        vis      = Vis_faces[s_u, s_v] & ~seen_u
        Seen[v]  = seen_u | vis

        # SIMULATION: estimate the reward from this new state
        distfac = np.exp(-lam*<double>(Dist[v] - Dist[T.root]))
        gain    = sum_where_1d(face_score, vis) * distfac
        score   = Score[v] = score_u + gain

        # Strict inequality ensures that if all nodes are equal (i.e. if no new
        # faces were found), we return the root node.
        if score > best_score:
            best_score = score
            best_node = v

        # PROPAGATION: bubble the reward up the tree
        N_child[v] = 0
        for w in path:
            N_child[w] = <int>N_child[w] + 1

        if N_child[T.root] >= max_size:
            break

        if step == 0 or ((step+1) % (steps//6)) == 0:
            best_path   = T.path(T.root, best_node)
            best_path_s = [S[u] for u in best_path]
            log.debug(f'{N_child[T.root]:5d} rollouts in {len(T):5d} nodes, score: {Score[best_node]:.5g}')
            gui.hilight_roadmap_edges(G, pairwise(best_path_s))
            gui.wait_draw()

    if not any([np.any(Seen[v] & ~Seen[T.root]) for v in T]):
        if best_node == T.root:
            log.warn('did not find any unseen faces')
        else:
            log.warn('did not find any unseen faces (but best node is not root?)')
            best_node, best_score = T.root, Score[T.root]
    elif best_node == T.root:
        log.warn('best node is root even though other nodes saw new faces')
        #best_node = min(T, key=lambda v: G_dists[S[v]])
        #best_score = scoref(best_node)

    T.graph['best_node']  = best_node
    T.graph['best_score'] = best_score

    log.info('run complete, stats follow')
    log.info('inserted %d (%d recurrent paths)', n_inserted, n_recurred)
    log.info('   dist: %s', statstr(list(Dist.values())))
    log.info('  depth: %s', statstr(list(Depth.values())))
    log.info('n_child: %s', statstr(list(N_child.values())))
    log.info(' degree: %s', statstr(list(map(G.degree, G))))
    log.info('balance: %s', statstr([(N_child[T.parent(v)]+1)/(N_child[v]+1) for v in T if v != T.root]))
    log.info('  score: %s', statstr(list(Score.values())))

    return T
