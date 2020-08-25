"Plan a path for exploration in a state graph"

import sys
import logging
import warnings

import numpy as np
from networkx.utils import pairwise

import gui
import parame
from tree import Tree, EdgeDataMap, NodeDataMap

try:
    import cts
except ImportError as e:
    warnings.warn(f'cts import failed: {e}')
    cts = None


log = logging.getLogger(__name__)
cfg = parame.Module(__name__)


# Label maker. The advantage of doing this is that we can never have a bug
# where a new node gets assigned an already existing node label. We might run
# out of node labels though, and they may get pretty long.
next_label = iter(range(sys.maxsize)).__next__


def new(*, start, seen):
    "Create a new search tree from given *start* and *seen* faces."
    r = next_label()
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


@parame.configurable
def expand(T, *, roadmap,
           max_depth: cfg.param = 50,
           steps:     cfg.param = 15000,
           max_size:  cfg.param = 2.00,
           lam:       cfg.param = 50e-2,
           min_faces: cfg.param = 1):

    if isinstance(max_size, float):
        max_size = int(steps*max_size)

    if cts is not None:
        return cts.expand(T, roadmap=roadmap, max_depth=max_depth, steps=steps,
                          max_size=max_size, lam=lam, min_faces=min_faces)

    log.warn('using pure python tree expansion')

    G = roadmap

    R          = NodeDataMap(G, 'r')
    D          = EdgeDataMap(G, 'd')
    Vis_faces  = EdgeDataMap(G, 'vis_faces')

    S          = NodeDataMap(T, 's')
    Seen       = NodeDataMap(T, 'seen')
    Score      = NodeDataMap(T, 'score')
    Dist       = NodeDataMap(T, 'dist')
    Depth      = NodeDataMap(T, 'depth')
    N_child    = NodeDataMap(T, 'n_child')
    Unvisited  = NodeDataMap(T, 'unvisited')

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

    gui.update_face_hsv(hues=0.8/face_score.max()*face_score)
    #gui.update_face_hsv(hues=0.8/face_degree.max()*face_degree)

    log.info('  face_area: %s', statstr(face_area[vis_unseen]))
    log.info('face_degree: %s', statstr(face_degree[vis_unseen]))
    log.info(' face_score: %s', statstr(face_score[vis_unseen]))
    log.info('          d: %s', statstr(list(D.values())))

    def select_path():
        # SELECTION: find an unexplored branching point in the tree. Result is
        # the chosen next state S[v], and its path.

        path, u = [T.root], T.root

        for i in range(max_depth):

            if u not in Unvisited:
                unvisited_u = Unvisited[u] = list(G.adj[S[u]])
            else:
                unvisited_u = Unvisited[u]

            if unvisited_u:
                # Create a new node v and assign it a random successor state
                v = next_label()
                T.add_edge(u, v)
                #S[v] = s_v = np.random.choice(unvisited_u)
                #unvisited_u.remove(s_v)
                S[v] = unvisited_u.pop()
                return path + [v]

            else:
                u = min(T.adj[u], key=N_child.__getitem__)
                path.append(u)

        log.warn('reached maximum depth, terminating at visited state')
        raise StopIteration

    path_gen = iter(select_path, None)

    log.info('expanding tree %d steps (initial size: %d, max size: %d)',
             steps, len(T), max_size)

    best_node  = T.graph['best_node']
    best_score = T.graph['best_score']

    for i, path in zip(range(steps), path_gen):

        # EXPANSION: add new node to search tree
        u, v     = path[-2:]
        s_u      = S[u]
        s_v      = S[v]
        vis      = Vis_faces[s_u, s_v] & ~Seen[u] & face_vis
        Dist[v]  = Dist[u]  + D[s_u, s_v]
        Depth[v] = Depth[u] + 1
        Seen[v]  = Seen[u] | vis

        # SIMULATION: estimate the reward from this new state
        gain  = np.sum(face_score, where=vis) * np.exp(-lam*(Dist[v] - Dist[T.root]))
        score = Score[v] = Score[u] + gain

        # Strict inequality ensures that if all nodes are equal (i.e. if no new
        # faces were found), we return the root node.
        if score > best_score:
            best_score = score
            best_node = v

        # PROPAGATION: bubble the reward up the tree
        for w in path + [v]:
            N_child[w] = N_child.get(w, 0) + 1

        if N_child[T.root] >= max_size:
            break

        # Prevent programmer mistakes
        del path, u, v, w, s_u, s_v

        if i == 0 or ((i+1) % (steps//6)) == 0:
            best_path   = T.path(T.root, best_node)
            best_path_s = [S[u] for u in best_path]
            log.debug(f'{N_child[T.root]:5d} rollouts in {len(T):5d} nodes, score: {Score[best_node]:.5g}')
            gui.hilight_roadmap_edges(G, pairwise(best_path_s))
            gui.wait_draw()

    if not any(np.any(Seen[v] & ~Seen[T.root]) for v in T):
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
    log.info('   dist: %s', statstr(list(Dist.values())))
    log.info('  depth: %s', statstr(list(Depth.values())))
    log.info('n_child: %s', statstr(list(N_child.values())))
    log.info(' degree: %s', statstr(list(map(G.degree, G))))
    log.info('balance: %s', statstr([N_child[T.parent(v)]/N_child[v] for v in T if v != T.root]))
    log.info('  score: %s', statstr(list(Score.values())))

    return T


def is_done(T):
    "Do we know some place to go?"
    return T.graph['best_node'] != T.root


def best_path(T):
    Seen   = NodeDataMap(T, 'seen')
    S      = NodeDataMap(T, 's')
    path   = T.path(T.root, T.graph['best_node'])
    path_s = [S[u] for u in path]
    return path_s, Seen[path[-1]]


@parame.configurable
def plan_path(*, start, seen, roadmap,
              num_expand: cfg.param = 10):

    stree = new(start=start, seen=seen)

    for i in range(num_expand):

        log.info('expanding search tree iteration %d', i)

        stree = expand(stree, roadmap=roadmap)

        if is_done(stree):
            break

    return best_path(stree)
