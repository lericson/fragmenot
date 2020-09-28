"Plan a path for exploration in a state graph"

import sys
import logging
import warnings

import numpy as np
from networkx.utils import pairwise

import gui
import cts
import parame
from tree import Tree, EdgeDataMap, NodeDataMap


class NoPlanFoundError(Exception):
    pass


log = logging.getLogger(__name__)
cfg = parame.Module(__name__)


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
           steps:          cfg.param = 15000,
           alpha:          cfg.param = 1e-1,
           lam:            cfg.param = 35e-2,
           K:              cfg.param = 1e3,
           score_function: cfg.param = 'ours',
           weight_function: cfg.param = 'ours',
           path_selection: cfg.param = 'ours'):

    weight_function = cts.weight_functions[weight_function]
    score_function  = cts.score_functions[score_function]
    path_selection  = cts.path_selections[path_selection]

    return cts.expand(T, roadmap=roadmap, steps=steps, alpha=alpha, lam=lam,
                      K=K, score_function=score_function,
                      weight_function=weight_function,
                      path_selection=path_selection)


def is_done(T):
    "Do we know some place to go?"
    return T.graph['best_node'] != T.root


@parame.configurable
def plan_path(*, start, seen, roadmap,
              num_expand: cfg.param = 3):

    stree = new(start=start, seen=seen)

    for i in range(num_expand):

        log.info('expanding search tree iteration %d', i)

        stree = expand(stree, roadmap=roadmap)

        if is_done(stree):
            break

    else:
        raise NoPlanFoundError()

    return stree.graph['best_path_G'], stree.graph['best_path_seen']
