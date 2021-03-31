# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3, warn.unused_result=True, warn.unused_arg=True
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
from libcpp.limits cimport numeric_limits

import numpy as np
cimport numpy as cnp
from networkx.utils import pairwise

import gui
import parame
from tree import Tree


log = logging.getLogger(__name__)

cfg = parame.Module(__name__)

cdef enum: SF_OURS, SF_GN
cdef enum: WF_OURS, WF_UNIFORM, WF_UNIFORM_COLORED
cdef enum: PS_OURS, PS_SHORTEST
score_functions  = {'ours': SF_OURS, 'gn': SF_GN}
weight_functions = {'ours': WF_OURS, 'uniform': WF_UNIFORM, 'colored_uniform': WF_UNIFORM_COLORED}
path_selections  = {'ours': PS_OURS, 'shortest': PS_SHORTEST}


class NoPlanFoundError(Exception):
    pass


def statstr(a):
    a = np.atleast_1d(a)
    if len(a) == 0:
        return 'N=0, min=?, mean=?, max=?, std=?'
    elif len(a) == 1:
        return f'N=1, min={a[0]:.5g}, mean={a[0]:.5g}, max={a[0]:.5g}, std=?'
    else:
        return ', '.join([f'N={len(a)}',
                          f'min={np.min(a):.5g}', f'mean={np.mean(a):.5g}',
                          f'max={np.max(a):.5g}', f'std={np.std(a):.5g}'])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double [::1] compute_covis_area(const unsigned char [:, :] covis,
                                     const unsigned char [:] vis,
                                     const double [:] area,
                                     double [::1] covis_area) nogil:
    cdef double area_i
    cdef double min_area
    with gil:
        min_area = cfg.get('min_area', 0.0)
        assert covis_area.shape[0] == covis.shape[0]
        assert area.shape[0]       == covis.shape[0]

    for i in range(covis.shape[0]):
        area_i = 0.0
        if vis[i]:
            for j in range(covis.shape[1]):
                if vis[j] and covis[i, j]:
                    area_i += area[j]

        if 0.0 < area_i < min_area:
            area_i += 200.0
        covis_area[i] = sqrt(area_i)

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
    int parent
    int s
    int depth
    double score
    double dist
    #unsigned char [::1]* seen
    int_list unvisited


cdef inline long h(int a, int b) nogil:
    "Hashing pairs of integers is hard for STL."
    return ((<long>(<unsigned int>a)) << 8*sizeof(int)) | <unsigned int>b


cdef cppclass EdgeExistsPredicate:
    const unordered_set[long] *E
    int s

    EdgeExistsPredicate(unordered_set[long] &E, int s) nogil:
        E  # Warns otherwise.
        this.E = &E
        this.s = s

    bint __call__ "operator()"(int v):
        return E.count(h(s, v)) > 0


def unit_map(a, *, zero, one):
    return zero + a*(one - zero)


def expand(object T, *, object roadmap):

    assert 2*sizeof(int) == sizeof(long), 'long is twice the size of int'

    cdef:
        # parame doesn't work in Cython, do config oldskool style.
        int    steps = cfg.get('steps', 30000)
        double alpha = cfg.get('alpha', 12e-1)
        double lam   = cfg.get('lam',   35e-2)
        double K     = cfg.get('K',     1e3)

        int weight_function = weight_functions[cfg.get('weight_function', 'ours')]
        int score_function  = score_functions[cfg.get('score_function', 'ours')]
        int path_selection  = path_selections[cfg.get('path_selection', 'ours')]

        int step, i, j, u, v, w

        object G = roadmap
        int F = T.nodes[T.root]['seen'].shape[0]
        int N = len(T)

        vector[int] intvec
        hashmap[long, double] D
        hashmap[long, unsigned char [::1]] Vis_faces
        hashmap[int, int_list] G_adj

        unsigned char [::1] vis

    # In shortest-path-style search, we visit each edge in G twice. Not sure why, but it
    # turns out that way.
    if path_selection == PS_SHORTEST:
        steps = 2*len(roadmap.edges) + 1

    D.reserve(len(G.edges))
    Vis_faces.reserve(len(G.edges))
    G_adj.reserve(len(G))

    # Copy G._adj to G_adj
    for i, adj in G._adj.items():
        G_adj[i] = adj
        for j, dd_ij in adj.items():
            vis = dd_ij['vis_faces']
            assert vis.shape[0] == F
            D[h(i, j)]         = D[h(j, i)]         = dd_ij['d']
            Vis_faces[h(i, j)] = Vis_faces[h(j, i)] = vis
            assert (i, j) in G.edges

    cdef:
        vector[int] path_T
        vector[int] path_G
        int root = 0

        unsigned char [:, ::1] Seen = np.empty((N + steps, F), dtype=np.bool)

        vector[NodeT] Node
        hashmap[int, int_list] T_succ

    assert T.root == root

    Node.resize(N)
    Node.reserve(N + steps)

    T_succ.reserve(N + steps)

    # Copy T to Node and T_succ.
    for i in range(N):
        T_succ[i] = T.succ[i]
        neighbors_i = set(G.adj[T.nodes[i]['s']])
        visited_i = {T.nodes[j]['s'] for j in T.succ[i]}
        dd = T.nodes[i]
        vis     = dd['seen']
        Seen[i] = vis
        parent = T.pred[i][0] if T.pred[i] else T.root
        Node[i] = NodeT(parent=parent, s=dd['s'], score=dd['score'],
                        dist=dd['dist'], depth=dd['depth'],
                        unvisited=(neighbors_i - visited_i))

    log.debug('computing weights')

    face_vis   = roadmap.graph['vis_faces']
    face_covis = roadmap.graph['covis_faces']
    face_area  = roadmap.graph['area_faces']

    vis_unseen = face_vis & ~np.asarray(Seen[root])

    # Compute face covisibility area, excluding seen faces.
    covis_area = np.empty(F, dtype=np.float)
    log.debug('computing covis area')
    if weight_function == WF_OURS or weight_function == WF_UNIFORM_COLORED:
        compute_covis_area(face_covis,
                           vis_unseen,
                           face_area,
                           covis_area)
    elif weight_function == WF_UNIFORM:
        covis_area[:] = 0.0
    else:
        raise ValueError(f'weight_function={weight_function} invalid')

    log.debug('computing face score')
    # 64-bit floats have a minimum exponent of 2^-1023 ~ 1e-308.
    face_score = K*np.exp(-alpha*covis_area)
    #face_score[~vis_unseen]             = K

    if weight_function != WF_UNIFORM:
        log.debug('updating face colors')
        gui.update_face_color(gui.color_envmesh, faces=Seen[root], layer='score')
        gui.update_face_colormap(((covis_area[vis_unseen]-5.0)/10.0).clip(0.05, 1.0),
                                 faces=vis_unseen, layer='score')
        #gui.update_face_colormap((-np.log(face_score[vis_unseen]/K)/16).clip(0.0, 1.0),
        #                         faces=vis_unseen, layer='score')

    if weight_function == WF_UNIFORM_COLORED:
        covis_area[:] = 0.0
        face_score[:] = K

    log.info('  face_area: %s', statstr(face_area[vis_unseen]))
    log.info(' covis_area: %s', statstr(covis_area[vis_unseen]))
    log.info(' face_score: %s', statstr(face_score[vis_unseen]))
    #log.info('          d: %s', statstr(list(D.values())))

    log.info('expanding tree %d steps')

    cdef:
        NodeT *node_u
        NodeT *node_v
        NodeT *node_w
        NodeT *node_root = &Node[root]

        int          best_node   = root
        vector[int]  best_path_T = [root]
        vector[int]  best_path_G = [node_root.s]
        double       best_score  = node_root.score

        double       gain_v
        double [::1] face_score_vw = face_score

        # For PS_SHORTEST, set of visited _EDGES_.
        unordered_set[int] visited = {node_root.s}
        NodeT node = NodeT(parent=0, s=0, depth=0, score=0.0, dist=0.0, unvisited=int_list())
        int search_start = 0

    assert node_root.dist == 0.0

    with nogil:

      for step in range(steps):

        # search_start the index of the first node with unvisited neighbors.
        for search_start in range(search_start, <int>Node.size()):
            if not Node[search_start].unvisited.empty():
                break

        # Find leaf with smallest d.
        d = numeric_limits[int].max()
        u = root
        for v in range(search_start, <int>Node.size()):
            #node_v = &Node[v]
            if not Node[v].unvisited.empty():
                if Node[v].depth < d:
                    u = v
                    d = Node[v].depth
                if Node[search_start].depth < Node[v].depth:
                    break

        # Search finished.
        if Node[u].unvisited.empty():
            break

        # Construct a new tree node.
        v = Node.size()
        Node.push_back(node)
        node_v = &Node[v]
        node_u = &Node[u]
        node_v.parent = u
        node_v.s = node_u.unvisited.front()
        node_u.unvisited.pop_front()
        T_succ[u].push_front(v)

        if path_selection == PS_OURS:
            node_v.unvisited = G_adj[node_v.s]

        elif path_selection == PS_SHORTEST:
            if not visited.count(node_v.s):
                node_v.unvisited = G_adj[node_v.s]
                visited.insert(node_v.s)
            else:
                node_v.unvisited.clear()

        # Set inserted node v's attributes
        node_v = &Node[v]
        node_v.dist  = node_u.dist + D[h(node_u.s, node_v.s)]
        node_v.depth = node_u.depth + 1

        gain_v = propagate(face_score_vw, Seen[u], Vis_faces[h(node_u.s, node_v.s)], Seen[v])

        if score_function == SF_OURS:
            node_v.score = node_u.score + gain_v * exp(-lam*node_v.dist)
        elif score_function == SF_GN:
            node_v.score = (node_u.score * node_u.dist + gain_v) / node_v.dist

        # Strict inequality ensures that if all nodes are equal (i.e. if no new
        # faces were found), we return the root node.
        if best_score < node_v.score:
            # Reconstruct path to this node.
            path_T.resize(node_v.depth + 1, -1)
            path_G.resize(node_v.depth + 1, -1)
            x = v
            while True:
                path_T[Node[x].depth] = x
                path_G[Node[x].depth] = Node[x].s
                if x == root:
                    break
                x = Node[x].parent
            best_score  = node_v.score
            best_path_T = path_T
            best_path_G = path_G
            best_node   = v

        if step == 0 or ((step+1) % (steps//6)) == 0:
          with gil:
            assert -1 not in list(best_path_T)
            log.debug(f'{Node.size():5d} nodes, score: {best_score:.5g}')
            gui.hilight_roadmap_edges(G, pairwise(best_path_G))

    if path_selection == PS_SHORTEST:
        missed = set(G.nodes) - set(visited)
        if missed:
            log.error('shortest-path search did not visit all nodes. missed: %s', missed)
            log.error('set(G.nodes)=%s, set(visited)=%s', set(G.nodes), set(visited))

        visited_edges  = [(Node[u].s, Node[v].s) for u, succ_u in T_succ for v in succ_u]
        visited_edges += [(Node[v].s, Node[u].s) for u, succ_u in T_succ for v in succ_u]
        missed = set(G.edges) - set(visited_edges)
        if missed:
            log.error('shortest-path did not visit all edges. missed: %s', missed)

    T.graph['best_score']     = best_score
    T.graph['best_path_T']    = list(best_path_T)
    T.graph['best_path_G']    = list(best_path_G)
    T.graph['best_node']      = best_node
    T.graph['best_path_seen'] = np.asarray(Seen[best_node])

    log.info('run complete, stats follow')

    log.info('search_start: %s', search_start)
    log.info('  depth: %s', statstr([n.depth     for n in Node]))
    log.info('   dist: %s', statstr([n.dist      for n in Node]))
    log.info('  score: %s', statstr([n.score     for n in Node]))
    log.info('odegree: %s', statstr([T_succ[i].size() for i in range(<int>Node.size())]))

    if path_selection == PS_OURS:
        # Print out stats on how deep search reached
        Num_vis = [0]*10
        Num_unv = [0]*10
        for i in range(<int>Node.size()):
            node_v = &Node[i]
            if len(Num_vis) <= node_v.depth:
                continue
            Num_vis[node_v.depth] += len(T_succ[i])
            Num_unv[node_v.depth] += len(node_v.unvisited)

        for depth, (num_vis, num_unv) in enumerate(zip(Num_vis, Num_unv)):
            if num_vis + num_unv == 0:
                break
            log.info('num visited + unvisited states at depth=%d:%7d + %d', depth, num_vis, num_unv)

    return T


def new(*, start, seen):
    "Create a new search tree from given *start* and *seen* faces."
    r = 0
    T = Tree()
    T.root = r
    T.add_node(r)
    T.nodes[r].update({'s':          start,
                       'seen':       seen.copy(),
                       'score':      0.0,
                       'dist':       0.0,
                       'depth':      0,
                       'n_child':    0})

    T.graph['best_node']  = r
    T.graph['best_score'] = -np.inf

    return T


def stump(T, r):
    "Create a new search tree from T with only node r"
    T_     = Tree()
    T_._node[r] = d = T._node[r].copy()
    T_.graph        = T.graph.copy()
    del T
    T_.root = r
    T_.graph['best_node']  = r
    T_.graph['best_score'] = d['score']
    T_._succ[r] = {}
    T_._pred[r] = {}
    del d['unvisited']
    d['n_child'] = 0
    return T_


def plan_path(*, start, seen, roadmap):
    stree = new(start=start, seen=seen)
    stree = expand(stree, roadmap=roadmap)
    if stree.graph['best_node'] == stree.root:
        raise NoPlanFoundError()
    return stree.graph['best_path_G'], stree.graph['best_path_seen']
