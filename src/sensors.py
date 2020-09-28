"""Sensor simulation

Su nombre no es Ray uno,
          no es Ray dos,
Su nombre    es Ray tres.
"""


import time
import logging
import collections
from os import path, makedirs

import numpy as np
import trimesh

import gui
import parame
from utils import graph_md5


cfg = parame.Module(__name__)
log = logging.getLogger(__name__)

_sensor_geometry   = trimesh.creation.icosphere(subdivisions=4)
_sensor_directions = _sensor_geometry.vertices


@parame.configurable
def visible_points(mesh, R, *,
                   radius:              cfg.param = 2.0,
                   max_incidence_angle: cfg.param = np.radians(85)):
    "Union of faces visible at each R[i]."

    directions = _sensor_directions
    min_dists  = np.full(mesh.faces.shape[0], np.inf)
    rays_face  = np.empty(directions.shape[0], dtype=np.int32)
    num_tested = 0
    cos_incid  = np.cos(max_incidence_angle)

    for origin in R:
        num_tested += len(directions)

        mesh.ray.intersects_first([origin], directions, max_dists=radius, out=rays_face)

        # Mask rays that hit
        rays_hit = rays_face >= 0

        # Consider only hits with incidence angle below threshold
        rays_dir   = directions[rays_hit]
        faces_norm = mesh.face_normals[rays_face[rays_hit]]
        rays_hit[rays_hit] = np.sum(rays_dir * faces_norm, axis=1) < cos_incid

        # Faces hit by a ray
        faces = rays_face[rays_hit]

        dists_hits = np.linalg.norm(origin - mesh.triangles_center[faces], axis=1)
        min_dists[faces] = np.min((min_dists[faces], dists_hits), axis=0)

    vis = min_dists < radius

    return vis, min_dists, num_tested


@parame.configurable
def sample_points(r_u, r_v, *, step_size: cfg.param = 0.5):
    dist   = np.linalg.norm(r_v - r_u)
    steps  = int(dist / step_size) + 1
    points = np.linspace(r_u, r_v, steps)
    assert np.allclose(points[0],  r_u), (dist, steps, points)
    assert np.allclose(points[-1], r_v), (dist, steps, points)
    return points


def _update_edge_visibility(G, mesh, *, force=False, seen=None):
    """Compute which faces are visible while traversing each edge in G

    Parameters
    ----------
    G : networkx.Graph
      Graph over which to compute, results are inserted as edge data
    mesh : trimesh.Trimesh
      Mesh to test visibility on
    force : bool
      Force updating the edge information even if it already exists
    seen : (n,) bool
      Ignore these faces when testing for visibility
    """

    N    = mesh.faces.shape[0]
    mask = ~seen if seen is not None else True

    # All visible faces. Useful to determine % explored.
    if 'vis_faces' not in G.graph:
        vis_faces = np.zeros(N, dtype=bool)
    else:
        vis_faces = G.graph['vis_faces']

    # Covisibility matrix. If M[i, j] != 0, then i is visible with j.
    if 'covis_faces' not in G.graph:
        log.debug('pre-allocating covisibility matrix')
        covis_faces = np.empty((N, N), dtype=np.bool)
        covis_faces[:, :] = False
    else:
        covis_faces = G.graph['covis_faces']

    edges = [(u, v, dd) for (u, v, dd) in G.edges.data()
             if force or 'vis_faces' not in dd]

    log.info(f'computing sensor visibility over {len(edges)} edges')

    # For printing statistics
    num_tested_sum = 0
    num_faces_sum  = 0
    ip             = 0
    tp             = time.time()

    for i, (u, v, data_uv) in enumerate(edges):

        r_u = G.nodes[u]['r']
        r_v = G.nodes[v]['r']

        vis_uv, dists_uv, num_tested = visible_points(mesh, sample_points(r_u, r_v))

        vis_uv               &= mask
        vis_faces            |= vis_uv
        covis_faces[vis_uv]  |= vis_uv
        data_uv['vis_faces']  = vis_uv
        data_uv['dist_faces'] = dists_uv

        num_tested_sum += num_tested
        num_faces_sum  += np.count_nonzero(vis_uv)

        if i > 0 and (i % (max(10, len(edges))//10)) == 0:
            t = time.time()
            gui.update_vis_faces(visible=vis_faces)
            log.debug(f'finished {i+1:5d}/{len(edges)}. '
                      f'tested {num_tested_sum/(i-ip):.2f} rays/edge. ' 
                      f'hit {num_faces_sum/(i-ip):.2f} faces/edge. '
                      f'speed {num_tested_sum/(t-tp):.5g} rays/sec, '
                      f'{(i-ip)/(t-tp):.5g} edges/sec.')
            num_tested_sum = 0
            num_faces_sum  = 0
            ip             = i
            tp             = time.time()

    G.graph['vis_faces']   = vis_faces
    G.graph['covis_faces'] = covis_faces
    G.graph['area_faces']  = mesh.area_faces


@parame.configurable
def update_edge_visibility(G, mesh, *, force=False, seen=None,
                           load_cache: cfg.param = True,
                           save_cache: cfg.param = True,
                           cache_path: cfg.param = './var/cache'):
    "Caching wrapper for _update_edge_visibility"

    cache_filename = f'edge_vis_{graph_md5(G)}_{_sensor_directions.md5()}_{mesh.md5()}.npz'
    cache_pathname = path.join(cache_path, cache_filename)

    if load_cache and path.exists(cache_pathname):
        log.info('loading edge visibility from cache at %s', cache_pathname)
        try:
            d = np.load(cache_pathname, allow_pickle=True)
            for uv, vis_uv, dists_uv in zip(d['edges'], d['vis_faces_edges'], d['dist_faces_edges']):
                G.edges[uv]['vis_faces']  = vis_uv
                G.edges[uv]['dist_faces'] = dists_uv
            G.graph['vis_faces']   = d['vis_faces']
            G.graph['covis_faces'] = d['covis_faces']
            G.graph['area_faces']  = mesh.area_faces
        except Exception as e:
            log.exception('failed loading cached edge visibility: %s', e)
        else:
            return

    if not trimesh.ray.has_embree:
        log.error('pyembree not in use!\n%s', trimesh.ray.ray_pyembree.exc)

    _update_edge_visibility(G, mesh, force=force, seen=seen)

    if save_cache:
        log.info('saving edge visibility to %s', cache_pathname)
        edges = np.array(G.edges, dtype=object)
        vis_faces_edges  = np.array([G.edges[uv]['vis_faces']  for uv in edges])
        dist_faces_edges = np.array([G.edges[uv]['dist_faces'] for uv in edges])
        makedirs(path.dirname(cache_pathname), exist_ok=True)
        d = dict(edges=edges,
                 vis_faces_edges=vis_faces_edges,
                 dist_faces_edges=dist_faces_edges,
                 vis_faces=G.graph['vis_faces'],
                 covis_faces=G.graph['covis_faces'])
        np.savez_compressed(cache_pathname, **d)


def update_vertex_visibility(G, mesh, *, root=None, seen=None):
    """Compute set of visible faces from each vertex in an acyclic digraph G by
    starting at root and propagating outwards.
    """
    log.info('computing accumulated visibility over branches')
    if seen is None:
        seen = np.zeros(mesh.faces.shape[0], dtype=bool)
    if root is None:
        root = G.root
    q = collections.deque([(root, seen.copy(), np.inf, -np.inf)])
    while q:
        u, vis_u, min_dist_u, max_dist_u = q.popleft()
        data_u = G.nodes[u]
        data_u['min_dist'] = min_dist_u
        data_u['max_dist'] = max_dist_u
        data_u['vis_faces'] = vis_u
        data_u['vis_area']  = np.sum(mesh.area_faces[vis_u])
        for u, v, data_uv in G.edges(u, data=True):
            vis_uv = data_uv['vis_faces']
            vis_v  = vis_u | vis_uv
            min_dist_v = np.min(data_uv['dist_faces'][vis_uv], initial=min_dist_u)
            max_dist_v = np.max(data_uv['dist_faces'][vis_uv], initial=max_dist_u)
            q.append((v, vis_v, min_dist_v, max_dist_v))


# Appendix A1

class MeshSensor(object):
    "Use a sphere mesh's vertices for raycasting"

    def __init__(self, *, max_distance=4.0, num_points=100):
        self.max_distance = max_distance
        self.mesh = trimesh.creation.icosphere()
        self.num_points = num_points
        self._points = np.empty_like(self.mesh.vertices)

    def visible_between(self, mesh, r_u, r_v):
        points = self._points
        verts = set()
        for point in np.linspace(r_u, r_v, self.num_points):
            points[:] = point
            hits = mesh.ray.intersects_first(points, self.mesh.vertices)
            faces = hits[hits >= 0]
            face_centroids = mesh.vertices[mesh.faces[faces]].mean(axis=1)
            dists = np.linalg.norm(point - face_centroids, axis=1)
            verts |= set(mesh.faces[faces[dists < self.max_distance]].flatten())
        return np.array(list(verts), dtype=int)
