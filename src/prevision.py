"""Deals with making the global roadmap a locally-aware one, with some given
distance of prevision.
"""


import logging

import numpy as np
from scipy.spatial import cKDTree

import parame
import sensors
from tree import NodeDataMap


log = logging.getLogger()
cfg = parame.Module(__name__)


@parame.configurable
def subgraph(roadmap, *, mesh, seen_faces, seen_states,
             max_distance: cfg.param = np.inf,
             use_cache:    cfg.param = True):

    log.info('creating prevision subgraph (d=%.2f)', max_distance)
    log.debug('seen_faces: %d, seen_states: %d',
              np.count_nonzero(seen_faces), len(seen_states))

    R              = NodeDataMap(roadmap, 'r')
    sensors_radius = sensors.visible_between.__kwdefaults__['radius']

    # Find the subset of states in roadmap that are within threshold distance
    # to seen_states.
    if use_cache and '_kd' in roadmap.graph:
        (states, kd_states) = roadmap.graph['_kd']
    else:
        states    = list(roadmap)
        kd_states = cKDTree([R[u] for u in states])
        roadmap.graph['_kd'] = (states, kd_states)

    # seen_states can have _a lot_ of duplicates.
    query_points = [R[u] for u in set(seen_states)]

    inds = kd_states.query_ball_point(query_points, r=max_distance + sensors_radius)

    states_aware = {states[j] for inds_i in inds for j in inds_i}

    log.debug('querying face neighbors')
    faces_center = mesh.triangles_center

    tree = cKDTree(faces_center[seen_faces])
    dists, inds = tree.query(faces_center, distance_upper_bound=max_distance)

    vis_faces  = roadmap.graph['vis_faces'].copy()
    vis_faces &= dists < max_distance + 1e-8

    log.debug('building subgraph')
    roadmap_local = roadmap.copy()
    roadmap_local.remove_nodes_from(set(roadmap_local) - states_aware)
    roadmap_local.graph['vis_faces'] = vis_faces

    return roadmap_local
