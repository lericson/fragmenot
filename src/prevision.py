"""Deals with making the global roadmap a locally-aware one, with some given
distance of prevision.
"""


import logging

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

import prm
import parame
import sensors
from tree import NodeDataMap


log = logging.getLogger()
cfg = parame.Module(__name__)


@parame.configurable
def subgraph(roadmap, *, mesh, seen_faces, seen_states,
             max_distance: cfg.param = np.inf):

    # seen_states can have _a lot_ of duplicates.
    seen_states = set(seen_states)

    log.info('creating prevision subgraph (d=%.2f)', max_distance)
    log.debug('seen_faces: %d, seen_states: %d',
              np.count_nonzero(seen_faces), len(seen_states))

    R              = NodeDataMap(roadmap, 'r')
    sensors_radius = sensors.visible_between.__kwdefaults__['radius']

    if '_kd' in roadmap.graph:
        (states, kd_states) = roadmap.graph['_kd']
    else:
        states    = list(roadmap)
        kd_states = cKDTree([R[u] for u in states])
        roadmap.graph['_kd'] = (states, kd_states)

    # Find the subset of states in roadmap that are within threshold distance
    # to seen_states.
    query_points = [R[u] for u in seen_states]

    inds = kd_states.query_ball_point(query_points, r=max_distance + sensors_radius)

    states_aware = {states[j] for inds_i in inds for j in inds_i}

    log.debug('querying neighboring faces')

    face_pts       = mesh.triangles_center

    if 'kd_faces' not in mesh._cache.cache:
        kd_faces = cKDTree(face_pts)
        mesh._cache.cache['kd_faces'] = kd_faces
    else:
        kd_faces = mesh._cache.cache['kd_faces']

    inds = kd_faces.query_ball_point(face_pts[seen_faces], r=max_distance)
    vis_faces = np.zeros_like(roadmap.graph['vis_faces'])
    for inds_i in inds:
        vis_faces[inds_i] = True

    log.debug('building subgraph')
    roadmap_local = roadmap.copy()
    roadmap_local.graph['vis_faces'] = vis_faces

    roadmap_local.remove_nodes_from(set(roadmap_local) - states_aware)
    comps = nx.connected_components(roadmap_local)
    reachable  = set.union(*[c for c in comps if set(c) & set(seen_states)])
    roadmap_local.remove_nodes_from(states_aware - reachable)

    return roadmap_local
