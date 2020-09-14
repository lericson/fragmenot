"""Deals with making the global roadmap a locally-aware one, with some given
distance of prevision.
"""


import logging

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

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

    # Build kd tree of roadmap states
    states    = list(roadmap)
    kd_states = cKDTree([R[u] for u in states])

    # Find subset of states that are within threshold distance to seen_states.
    qry_pts = [R[u] for u in seen_states]
    inds = kd_states.query_ball_point(qry_pts, r=max_distance + sensors_radius)
    states_aware = {states[j] for inds_i in inds for j in inds_i}

    log.debug('querying neighboring faces')
    face_pts    = mesh.triangles_center
    kd_faces    = cKDTree(face_pts[seen_faces])
    dists, inds = kd_faces.query(face_pts, distance_upper_bound=max_distance)
    vis_faces   = dists < max_distance

    log.debug('building subgraph')
    roadmap_local = roadmap.copy()
    roadmap_local.graph['vis_faces'] = vis_faces

    roadmap_local.remove_nodes_from(set(roadmap_local) - states_aware)
    comps = nx.connected_components(roadmap_local)
    reachable  = set.union(*[c for c in comps if set(c) & set(seen_states)])
    roadmap_local.remove_nodes_from(states_aware - reachable)

    return roadmap_local
