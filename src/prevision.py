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
             max_distance: cfg.param = np.inf):

    log.info('creating prevision subgraph (d=%.2f)', max_distance)

    R              = NodeDataMap(roadmap, 'r')
    sensors_radius = sensors.visible_between.__kwdefaults__['radius']

    tree   = cKDTree(np.array(seen_states))
    states = list(roadmap)
    poses  = np.array([R[u] for u in states])
    dists, inds = tree.query(poses, distance_upper_bound=max_distance + sensors_radius)

    states_unaware = [s for s, d in zip(states, dists) if max_distance + sensors_radius < d]

    log.debug('querying face neighbors')
    faces_center = mesh.triangles_center

    tree = cKDTree(faces_center[seen_faces])
    dists, inds = tree.query(faces_center, distance_upper_bound=max_distance)

    vis_faces  = roadmap.graph['vis_faces'].copy()
    vis_faces &= dists < max_distance + 1e-8

    log.debug('building subgraph')
    roadmap_local = roadmap.copy()
    roadmap_local.remove_nodes_from(states_unaware)
    roadmap_local.graph['vis_faces'] = vis_faces

    return roadmap_local
