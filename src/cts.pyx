# distutils: language = c++
# cython: language_level = 3
"Plan a path for exploration in a state graph"

import sys
cimport cython
import logging
from libc.math cimport exp
from libcpp.vector cimport vector
from libcpp.map cimport map as hashmap

import numpy as np
from networkx.utils import pairwise

import gui


log = logging.getLogger(__name__)


"""
# Label maker. The advantage of doing this is that we can never have a bug
# where a new node gets assigned an already existing node label. We might run
# out of node labels though, and they may get pretty long.
cdef int _label = 0
cdef inline int next_label() nogil:
    global _label
    _label += 1
    return _label
"""


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
cdef bint recurred(double [::1] Score, list path, list path_s, int state, double score):
    cdef int i
    for i in range(len(path) - 1):
        if path_s[i] == state and score <= Score[<int>path[i]]:
            return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _propagate(double [::1] face_score,
                       unsigned char [::1] seen_u,
                       unsigned char [::1] vis_uv,
                       unsigned char [::1] seen_v) nogil:
    "Compute seen_v, gain_v"
    cdef Py_ssize_t k = face_score.shape[0]
    cdef double gain_v = 0.0
    for i in range(k):
        seen_v[i] = seen_u[i] or vis_uv[i]
        if seen_v[i] and not seen_u[i]:
            gain_v += face_score[i]
    return gain_v


# cts.expand() is called by tree_search.expand(), which sets the defaults for
# all parameters.
def expand(object T, *, object roadmap,
           int    max_depth,
           int    steps,
           int    max_size,
           double alpha,
           double lam,
           double K):

    cdef object G = roadmap

    cdef EdgeDataMap D          = EdgeDataMap(G, 'd')
    cdef EdgeDataMap Vis_faces  = EdgeDataMap(G, 'vis_faces')

    cdef int step, i, j, u, v, w, s_u, s_v
    cdef list path, path_s
    cdef vector[int]* unvisited_u
    cdef int root = 0;
    cdef int N = len(T)
    cdef int next_label = N

    assert T.root == root

    cdef    int [::1] S       = np.empty(N + steps, dtype=np.int32)
    cdef double [::1] Score   = np.empty(N + steps, dtype=np.float64)
    cdef double [::1] Dist    = np.empty(N + steps, dtype=np.float64)
    cdef    int [::1] Depth   = np.empty(N + steps, dtype=np.int32)
    cdef    int [::1] N_child = np.empty(N + steps, dtype=np.int32)
    #cdef NodeDataMap Score      = NodeDataMap(T, 'score')
    #cdef NodeDataMap Dist       = NodeDataMap(T, 'dist')
    #cdef NodeDataMap Depth      = NodeDataMap(T, 'depth')
    #cdef NodeDataMap N_child    = NodeDataMap(T, 'n_child')
    #cdef dict Unvisited = {}
    cdef dict Seen      = {}

    cdef hashmap[int, vector[int]] Unvisited

    for i in range(N):
        dd         = T.nodes[i]
        S[i]       = dd['s']
        Seen[i]    = dd['seen']
        Score[i]   = dd['score']
        Dist[i]    = dd['dist']
        Depth[i]   = dd['depth']
        N_child[i] = dd['n_child']

    log.info('computing weights')

    face_vis   = roadmap.graph['vis_faces']
    face_covis = roadmap.graph['covis_faces']
    face_area  = roadmap.graph['area_faces']

    vis_unseen = face_vis & ~Seen[root]

    # Compute face covisibility area, excluding seen faces. We do this at a
    # smaller length scale because of integer mathematics. Do not change to
    # floats, it will waste vast amounts of memory.
    face_degree = face_covis.dot((1e5*face_area*vis_unseen).astype(np.uint32))/1e5
    face_degree[~face_vis] = 0

    #for (i,) in zip(*np.nonzero(face_vis)):
    #    assert (face_degree[i] == np.dot(face_covis[i], vis_unseen)), f'{i}: {face_degree[i]}'

    # Clip to 300 as a 64-bit float has a minimum exponent of 2^-1023 ~ 1e-308.
    face_score = K*np.exp(-alpha*(face_degree.clip(0, 200)))
    #face_score[face_degree < min_faces] = 1e-16
    face_score[~vis_unseen]             = K

    gui.update_face_hsv(hues=0.8*(1 - (-np.log(face_score/K)/alpha)/200), layer='score')
    #gui.update_face_hsv(hues=0.8/face_degree.max()*face_degree)

    log.info('  face_area: %s', statstr(face_area[vis_unseen]))
    log.info('face_degree: %s', statstr(face_degree[vis_unseen]))
    log.info(' face_score: %s', statstr(face_score[vis_unseen]))
    log.info('          d: %s', statstr(list(D.values())))

    log.info('expanding tree %d steps (initial size: %d, max size: %d)',
             steps, len(T), max_size)

    cdef int    best_node  = T.graph['best_node']
    cdef double best_score = T.graph['best_score']
    cdef dict Gadj = G._adj
    cdef dict Tadj = T._adj
    cdef object N_child_get = N_child.__getitem__
    cdef double score_u, score_v, gain_v, distfac_v, dist_u, dist_v
    cdef double dist_root = Dist[root]
    cdef double [::1] face_score_vw = face_score;

    cdef int n_recurred = 0, n_inserted = 0

    for step in range(steps):

        ## SELECTION: find an unexplored branching point in the tree. Result is
        ## the chosen next state S[v], and its path.

        path, u = [], root
        path_s = []

        # This loop generates a path = [..., u], and a successor state s_v of
        # s_u according to G.
        for j in range(max_depth):

            s_u = S[u]
            path.append(u)
            path_s.append(s_u)

            #if u not in Unvisited:
            if Unvisited.find(u) == Unvisited.end():
                Unvisited[u] = list(Gadj[s_u])
                unvisited_u = &Unvisited[u]
            else:
                unvisited_u = &Unvisited[u]

            if not unvisited_u.empty():
                # The tree node has unvisited neighboring states, let s_v be
                # one of them.
                s_v = unvisited_u.back()
                unvisited_u.pop_back()
                break

            nbrs = list(Tadj[u])
            if not nbrs:
                raise RuntimeError('node is leaf and no neighbor states')
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
             #continue
        n_inserted += 1

        # EXPANSION: add new node to search tree
        v = next_label
        next_label += 1
        S[v] = s_v
        T.add_node(v)
        T.add_edge(u, v)

        seen_v  = np.empty(face_score.shape, dtype=bool)
        gain_v  = _propagate(face_score_vw, Seen[u], Vis_faces[s_u, s_v], seen_v)
        Seen[v] = seen_v

        dist_u = Dist[u]
        dist_v = Dist[v] = dist_u + D[s_u, s_v]

        Depth[v] = <int>Depth[u] + 1

        # SIMULATION: estimate the reward from this new state
        distfac_v = exp(lam*(dist_root - dist_v))
        score_v   = Score[v] = score_u + gain_v * distfac_v

        # Strict inequality ensures that if all nodes are equal (i.e. if no new
        # faces were found), we return the root node.
        if score_v > best_score:
            best_score = score_v
            best_node = v

        # PROPAGATION: bubble the reward up the tree
        N_child[v] = 0
        for w in path:
            N_child[w] = <int>N_child[w] + 1

        if N_child[root] >= max_size:
            break

        if step == 0 or ((step+1) % (steps//6)) == 0:
            best_path   = T.path(root, best_node)
            best_path_s = [S[u] for u in best_path]
            log.debug(f'{N_child[root]:5d} rollouts in {len(T):5d} nodes, score: {Score[best_node]:.5g}')
            gui.hilight_roadmap_edges(G, pairwise(best_path_s))
            gui.wait_draw()

    if not any([np.any(Seen[v] & ~Seen[root]) for v in T]):
        if best_node == root:
            log.warn('did not find any unseen faces')
        else:
            log.warn('did not find any unseen faces (but best node is not root?)')
            best_node, best_score = root, Score[root]
    elif best_node == root:
        log.warn('best node is root even though other nodes saw new faces')
        #best_node = min(T, key=lambda v: G_dists[S[v]])
        #best_score = scoref(best_node)

    T.graph['best_node']  = best_node
    T.graph['best_score'] = best_score

    log.info('run complete, stats follow')
    log.info('inserted %d (%d recurrent paths)', n_inserted, n_recurred)
    log.info('   dist: %s', statstr(list(Dist)))
    log.info('  depth: %s', statstr(list(Depth)))
    log.info('n_child: %s', statstr(list(N_child)))
    log.info(' degree: %s', statstr(list(map(G.degree, G))))
    log.info('balance: %s', statstr([(N_child[T.parent(v)]+1)/(N_child[v]+1) for v in T if v != root]))
    log.info('  score: %s', statstr(list(Score)))

    T.S       = S
    T.Seen    = Seen
    T.Score   = Score
    T.Dist    = Dist
    T.Depth   = Depth
    T.N_child = N_child

    return T
