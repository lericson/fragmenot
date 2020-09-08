# distutils: language = c++
# cython: language_level = 3
"Plan a path for exploration in a state graph"

import sys
cimport cython
from cython.view cimport array as cvarray
#from cython.parallel import prange
from cython.operator cimport dereference as deref, preincrement as inc
import logging
from libc.math cimport exp, sqrt
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.list cimport list as stlist
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map as hashmap

import numpy as np
from networkx.utils import pairwise

import gui


# Sanity test that is quite expensive.
DEF test_seen = False

log = logging.getLogger(__name__)


cdef enum:
    SF_OURS
    SF_GN

score_functions = {'ours': SF_OURS,
                   'gn': SF_GN}

cdef enum:
    WF_OURS
    WF_UNIFORM

weight_functions = {'ours': WF_OURS,
                    'uniform': WF_UNIFORM}

cdef enum:
    PS_OURS
    PS_BFS

path_selections = {'ours': PS_OURS,
                   'bfs': PS_BFS}


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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double [::1] compute_covis_area(const unsigned char [:, :] covis,
                                     const unsigned char [:] vis,
                                     const double [:] area,
                                     double [::1] covis_area):
    cdef double area_i

    with nogil:
        for i in range(covis.shape[0]):
            area_i = 0.0
            if vis[i]:
                for j in range(covis.shape[1]):
                    if vis[j] and covis[i, j]:
                        area_i += area[j]

            covis_area[i] = 12*sqrt(area_i)

    return covis_area



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double propagate(double [::1] face_score,
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


ctypedef stlist[int] int_list
ctypedef stlist[int].iterator int_list_iterator
ctypedef stlist[int].const_iterator int_list_const_iterator


cdef struct NodeT:
    int s
    int depth
    int n_child
    bint skip
    double score
    double dist
    #unsigned char [::1]* seen
    int_list unvisited


cdef inline long h(int a, int b) nogil:
    "Hashing pairs of integers is hard for STL."
    return ((<long>(<unsigned int>a)) << 8*sizeof(int)) | <unsigned int>b


cdef cppclass EdgeExistsPredicate:
    unordered_set[long] *E
    int s

    EdgeExistsPredicate(unordered_set[long] &E, int s) nogil:
        this.E = &E
        this.s = s

    bint __call__ "operator()"(int v):
        return E.count(h(s, v)) > 0


# cts.expand() is called by tree_search.expand(), which sets the defaults for
# all parameters.
def expand(object T, *, object roadmap,
           int    max_depth,
           int    steps,
           int    max_size,
           double alpha,
           double lam,
           double K,
           int    weight_function,
           int    score_function,
           int    path_selection):

    cdef int step, i, j, u, v, w

    cdef object G = roadmap
    cdef int F = T.nodes[T.root]['seen'].shape[0]
    cdef int N = len(T)

    assert 2*sizeof(int) <= sizeof(long)
    cdef vector[int] intvec
    cdef hashmap[long, double] D
    cdef hashmap[long, unsigned char [::1]] Vis_faces
    cdef hashmap[int, int_list] G_adj

    cdef unsigned char [::1] vis

    D.reserve(len(G.edges))
    Vis_faces.reserve(len(G.edges))
    G_adj.reserve(len(G))

    for i, adj in G._adj.items():
        G_adj[i] = adj
        for j, dd_ij in adj.items():
            vis = dd_ij['vis_faces']
            assert vis.shape[0] == F
            D[h(i, j)]         = D[h(j, i)]         = dd_ij['d']
            Vis_faces[h(i, j)] = Vis_faces[h(j, i)] = vis
            assert (i, j) in G.edges

    cdef vector[int] path_T
    cdef vector[int] path_G
    cdef int root = 0

    assert T.root == root

    #cdef    int [::1] S       = np.empty(N + steps, dtype=np.intc)
    #cdef double [::1] Score   = np.empty(N + steps, dtype=np.float64)
    #cdef double [::1] Dist    = np.empty(N + steps, dtype=np.float64)
    #cdef    int [::1] Depth   = np.empty(N + steps, dtype=np.intc)
    #cdef    int [::1] N_child = np.empty(N + steps, dtype=np.intc)
    cdef unsigned char [:, ::1] Seen = np.empty((N + steps, F), dtype=np.bool)

    #cdef hashmap[int, int_list] Unvisited
    cdef vector[NodeT] Node

    cdef hashmap[int, int_list] T_succ

    Node.resize(N)
    Node.reserve(N + steps)

    T_succ.reserve(N + steps)

    for i in range(N):
        T_succ[i] = T.succ[i]
        neighbors_i = set(G.adj[T.nodes[i]['s']])
        visited_i = {T.nodes[j]['s'] for j in T.succ[i]}
        dd = T.nodes[i]
        vis     = dd['seen']
        Seen[i] = vis
        Node[i] = NodeT(s=dd['s'], score=dd['score'], dist=dd['dist'],
                        depth=dd['depth'], n_child=dd['n_child'], skip=False,
                        unvisited=(neighbors_i - visited_i))

    log.debug('computing weights')

    face_vis   = roadmap.graph['vis_faces']
    face_covis = roadmap.graph['covis_faces']
    face_area  = roadmap.graph['area_faces']

    vis_unseen = face_vis & ~np.asarray(Seen[root])

    # Compute face covisibility area, excluding seen faces. We do this at a
    # smaller length scale because of integer mathematics.
    covis_area = np.empty(F, dtype=np.float)
    log.debug('computing covis area')
    if weight_function == WF_OURS:
        compute_covis_area(face_covis,
                           vis_unseen,
                           face_area,
                           covis_area)
    elif weight_function == WF_UNIFORM:
        covis_area[:] = 0.0
    else:
        raise ValueError(f'weight_function={weight_function} invalid')

    log.debug('computing face score')
    # Clip to 300 as a 64-bit float has a minimum exponent of 2^-1023 ~ 1e-308.
    face_score = K*np.exp(-alpha*(covis_area.clip(0, 200)))
    #face_score[~vis_unseen]             = K

    log.debug('updating face colors')
    gui.update_face_hsv(hues=0.8*(1 - (-np.log(face_score/K)/alpha)/200), layer='score')
    #gui.update_face_hsv(hues=0.8/covis_area.max()*covis_area)

    log.info('  face_area: %s', statstr(face_area[vis_unseen]))
    log.info(' covis_area: %s', statstr(covis_area[vis_unseen]))
    log.info(' face_score: %s', statstr(face_score[vis_unseen]))
    #log.info('          d: %s', statstr(list(D.values())))

    log.info('expanding tree %d steps (initial size: %d, max size: %d)',
             steps, len(T), max_size)

    cdef NodeT *node_u
    cdef NodeT *node_v
    cdef NodeT *node_w
    cdef NodeT *node_root = &Node[root]

    cdef int         best_node   = root
    cdef vector[int] best_path_T = [root]
    cdef vector[int] best_path_G = [node_root.s]
    cdef double      best_score  = node_root.score
    cdef double gain_v
    cdef double [::1] face_score_vw = face_score;

    # For PS_BFS, set of visited _EDGES_.
    cdef unordered_set[long] visited
    cdef NodeT node = NodeT(0, 0, 0, False, 0.0, 0.0, int_list())

    cdef int_list_iterator it

    assert node_root.dist == 0.0

    for step in range(steps):

      with nogil:

        # Find a leaf.
        u = root
        path_T.clear()
        path_G.clear()

        # This loop generates a path_T = [..., u], and a successor state s_v of
        # s_u according to G.
        while not node_root.skip:

            node_u = &Node[u]
            path_T.push_back(u)
            path_G.push_back(node_u.s)

            #print(path_T, path_G)

            if T_succ.find(u) == T_succ.end():
                node_u.unvisited = G_adj[node_u.s]

            if path_selection == PS_BFS:
              node_u.unvisited.remove_if(EdgeExistsPredicate(visited, node_u.s))
              #node_u.unvisited.erase(remove_if(node_u.unvisited.begin(),
              #                                 node_u.unvisited.end(),
              #                                 EdgeExistsPredicate(visited, node_u.s)),
              #                       node_u.unvisited.end())

            if not node_u.unvisited.empty():
                # The tree node has unvisited neighboring states, create a new
                # node whose state is one of them.
                v = Node.size()
                Node.push_back(node)
                node_v   = &Node[v]
                node_v.s = node_u.unvisited.front()
                node_u.unvisited.pop_front()
                T_succ[u].push_front(v)
                if path_selection == PS_BFS:
                    visited.insert(h(node_u.s, node_v.s))
                    visited.insert(h(node_v.s, node_u.s))
                #print('insert', f'{v} (s={node_v.s})')
                break

            # Find the least visited successor of u in T. This results in
            # BFS-like behavior.
            with cython.boundscheck(False), cython.wraparound(False):
                v      = u
                node_v = &Node[v]
                for w in T_succ[u]:
                    node_w = &Node[w]
                    #print(' ', f'(node_w.n_child := {node_w.n_child}) < (node_v.n_child := {node_v.n_child})')
                    if not node_w.skip and node_w.n_child < node_v.n_child:
                        v      = w
                        node_v = &Node[v]

            if v == u:
                # u has no unvisited next states in G, and all its successors
                # in T are unvisitable. Mark u as unvisitable too.
                node_u.skip = True

                # Undo the two push_back()s at the beginning of this iteration
                path_T.pop_back()
                path_G.pop_back()

                # Test if empty in case we marked the root unvisitable.
                if not path_T.empty():
                    # Take the last state in the path
                    u = path_T.back()

                    # Undo its push_back()s too, as they will be on top
                    path_T.pop_back()
                    path_G.pop_back()

            else:
                # We found a successor state v of u, continue path from v.
                u = v

        else:
          with gil:
            # If we get here, the root is marked skip. The only valid reason
            # for this to happen is that we must have visited every edge.
            missed = {(u, v) for (u, v) in G.edges if visited.count(h(u, v)) == 0}
            if missed:
                log.error('internal inconsistency detected! root node marked '
                          'skip before visiting all edges. missed edges: %s',
                          missed)
            break

        # Set inserted node v's attributes
        node_v = &Node[v]
        node_v.dist  = node_u.dist + D[h(node_u.s, node_v.s)]
        node_v.depth = node_u.depth + 1

        gain_v = propagate(face_score_vw, Seen[u], Vis_faces[h(node_u.s, node_v.s)], Seen[v])

        if score_function == SF_OURS:
            node_v.score = node_u.score + gain_v * exp(-lam*node_v.dist)
        elif score_function == SF_GN:
            node_v.score = (node_u.score * node_u.dist + gain_v) / node_v.dist

        for w in path_T:
            Node[w].n_child += 1

        # Strict inequality ensures that if all nodes are equal (i.e. if no new
        # faces were found), we return the root node.
        if best_score < node_v.score:
            best_score = node_v.score
            best_path_T = path_T
            best_path_T.push_back(v)
            best_path_G = path_G
            best_path_G.push_back(node_v.s)
            best_node = v

        if node_root.n_child >= max_size:
            break

        if step == 0 or ((step+1) % (steps//6)) == 0:
            with gil:
                log.debug(f'{node_root.n_child:5d} rollouts in {Node.size():5d} nodes, score: {best_score:.5g}')
                gui.hilight_roadmap_edges(G, pairwise(best_path_G))
                gui.wait_draw()

    IF test_seen:
        if not any([np.any(np.asarray(Seen[v]) & ~np.asarray(Seen[root])) for v in range(<int>Node.size())]):
            if best_node == root:
                log.warn('did not find any unseen faces')
            else:
                log.warn('did not find any unseen faces (but best node is not root?)')
        elif best_node == root:
            log.warn('best node is root even though other nodes saw new faces')
            #best_node = min(T, key=lambda v: G_dists[S[v]])
            #best_score = scoref(best_node)

    T.graph['best_score']     = best_score
    T.graph['best_path_T']    = list(best_path_T)
    T.graph['best_path_G']    = list(best_path_G)
    T.graph['best_node']      = best_node
    T.graph['best_path_seen'] = np.asarray(Seen[best_node])

    log.info('run complete, stats follow')

    log.info('  depth: %s', statstr([n.depth     for n in Node]))
    log.info('   dist: %s', statstr([n.dist      for n in Node]))
    log.info('n_child: %s', statstr([n.n_child   for n in Node]))
    log.info('  score: %s', statstr([n.score     for n in Node]))
    log.info(' degree: %s', statstr([p.second.size() for p in T_succ]))
    #log.info('  depth: %s', statstr(list(Depth)))
    #log.info('n_child: %s', statstr(list(N_child)))
    #log.info(' degree: %s', statstr(list(map(G.degree, G))))
    #log.info('balance: %s', statstr([(N_child[T.parent(v)]+1)/(N_child[v]+1) for v in T if v != root]))
    #log.info('  score: %s', statstr(list(Score)))

    return T
