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


@parame.configurable
def visible_between_DEPRECATED(mesh, r_u, r_v, *, mask=None,
                    radius: cfg.param = 2.0,
                    enable_occlusions: cfg.param = True,
                    max_incidence_angle: cfg.param = np.cos(np.radians(90 + 1e-3))):
    """Compute faces visible to the sensor as it moves from r_u to r_v.

    1. Find faces whose center is less than the sensor radius away; these are
       candidates in the later occlusion testing.

       This is done by projecting all points (the face center) onto the line
       defined by r_u and r_v. First, we move all points so that the line
       starts at the origin (r_u). The locus t for a point is a scalar
       corresponding to the length on the line the point is closest to, i.e.,
       its least squares projection onto the line. It is computed by the dot
       product of the point and the line's unit direction vector n.

       The line segment between r_u and r_v is thus 0 < t < |r_v - r_u|. We now
       aim to find points whose distance to the line is less than the sensor
       reading. To find that distance, first compute where the closest sensor
       position is by clipping t to the line segment, then the position is
       r_u + tn.

    2. Perform backface culling on the set of candidates. This is done by
       projecting the ray onto the face normal. If it is positive, or only
       slightly negative, the ray is at best parallel to the face.

    3. Given our set of candidate points, compute occlusion by ray tracing from
       each sensor position as explained above in the direction of the point.
       Then aggregate all faces that were hit; this is then the sensor reading.

    Note that this method is not accurate: some points may only be visible at
    an angle, however, we only test visibility perpendicular to the line.  The
    actual solution is prohibitively complex and expensive to compute and would
    require vast amounts of ray tracing.
    """

    if mask is None:
        mask = np.ones(mesh.faces.shape[0], dtype=bool)
    else:
        mask = mask.astype(bool)

    # Set of candidate points.
    R_p   = mesh.triangles_center

    # Transform points to a system where line starts at the origin.
    R_rel = R_p - r_u

    # Line parameters.
    n     = r_v - r_u
    t1    = np.linalg.norm(n)
    n    /= t1

    # Loci of points.
    ts      = np.dot(R_rel, n)
    ts      = ts[:, np.newaxis]

    # Clip to nearest point on line segment.
    ts_clip = np.clip(ts, 0, t1)

    # Compute rays, their origin and lengths
    ts_clip_n = ts_clip*n
    rays    = R_rel - ts_clip_n
    origins = r_u + ts_clip_n
    dists   = np.linalg.norm(rays, axis=1)

    # Mask set of points close enough to the sensor _anywhere on the line_
    mask &= (dists < radius) 

    # Transform loci to [0, 1], plus minus radius. Note that this is not
    # clipped; ts_ is used for deciding which points to include, and not for
    # computing the ray direction or origin.
    ts_     = (np.squeeze(ts) + radius) / (t1 + 2*radius)

    # Mask set of points on the pertinent line segment
    mask &= (0 < ts_)
    mask &= (ts_ < 1.0) 

    """
    # Compute cosine of the ray incidence angle by the identity
    #   dot(a, b) = |a|*|b|*cos(theta) <=> cos(theta) = dot(a,b)/|a|/|b|
    # We already have norms for the rays, and the normals have unit length.
    ray_face_cos = np.sum(rays*mesh.face_normals, axis=1)/dists

    # Mask faces inside maximum incidence angle
    mask &= (ray_face_cos < max_incidence_angle)
    """

    num_tested = np.count_nonzero(mask)

    if enable_occlusions:
        origins_mask = origins[mask]
        rays_mask    = rays[mask]

        # hits is a vector of a face index or -1 for each ray.
        hits = mesh.ray.intersects_first(origins_mask, rays_mask)

        # hit_faces is a vector of indices to every face hit by a ray.
        hit_faces    = hits[hits >= 0]

        rays_traced  = origins_mask[hits >= 0] - R_p[hit_faces]
        dists_traced = np.linalg.norm(rays_traced, axis=1)
        faces        = hit_faces[dists_traced < radius]

        # Set only the coincided face indices to true
        mask[:]      = False
        mask[faces]  = True

    return mask, dists, num_tested


_sensor_geometry = trimesh.creation.icosphere(subdivisions=4)
_sensor_directions = _sensor_geometry.vertices
_sensor_origins = np.empty_like(_sensor_directions)

@parame.configurable
def visible_between(mesh, r_u, r_v, *, mask,
                    radius: cfg.param = 2.0,
                    step_size: cfg.param = 0.5):
    F = mesh.faces.shape[0]
    if mask is None:
        mask = np.full(F, True)
    else:
        assert mask.shape == (F,)
        mask = mask.astype(bool)

    directions = _sensor_directions
    origins    = _sensor_origins
    min_dists  = np.full(F, np.inf)
    steps      = int(np.linalg.norm(r_v - r_u) / step_size) + 1

    num_tested = 0
    for origin in np.linspace(r_u, r_v, steps):
        num_tested += len(directions)
        origins[:, :] = origin
        hits = mesh.ray.intersects_first(origins, directions, dists=radius)
        hits = hits[hits >= 0]
        dists_hits = np.linalg.norm(origin - mesh.triangles_center[hits], axis=1)
        min_dists[hits] = np.min((min_dists[hits], dists_hits), axis=0)

    return mask & (min_dists < radius), min_dists, num_tested


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
    mask = ~seen if seen is not None else None

    # All visible faces. Useful to determine % explored.
    if 'vis_faces' not in G.graph:
        vis_faces = np.zeros(N, dtype=bool)
    else:
        vis_faces = G.graph['vis_faces']

    # Covisibility matrix. If M[i, j] != 0, then i is visible with j.
    # 
    # We store it as a 64-bit integer matrix since it is used in a matrix
    # multiplication to compute covisibility degree, where it has to be.
    if 'covis_faces' not in G.graph:
        log.debug('pre-allocating covisibility matrix')
        covis_faces = np.empty((N, N), dtype=np.uint32)
        covis_faces[:, :] = False
    else:
        covis_faces = G.graph['covis_faces']

    # For printing statistics
    num_tested_sum = 0
    num_faces_sum  = 0
    t              = time.time()

    num_edges = len(G.edges())
    log.info(f'computing sensor visibility over {num_edges} edges')

    for i, (u, v, data_uv) in enumerate(G.edges.data()):

        if 'vis_faces' in data_uv and not force:
            continue

        r_u = G.nodes[u]['r']
        r_v = G.nodes[v]['r']

        vis_uv, dists_uv, num_tested = visible_between(mesh, r_u, r_v, mask=mask)

        vis_faces            |= vis_uv
        covis_faces[vis_uv]  |= vis_uv
        data_uv['vis_faces']  = vis_uv
        data_uv['dist_faces'] = dists_uv

        num_tested_sum += num_tested
        num_faces_sum  += np.count_nonzero(vis_uv)

        if i > 0 and (i % (max(10, num_edges)//10)) == 0:
            te = time.time()
            gui.update_vis_faces(visible=vis_faces)
            log.debug(f'computed sensor reading for edge {i:5d}/{num_edges}. '
                      f'tested {10*num_tested_sum/num_edges:.2f} rays/edge. ' 
                      f'hit {10*num_faces_sum/num_edges:.2f} faces/edge. '
                      f'speed {num_tested_sum/(te-t):.5g} rays/sec.')
            num_tested_sum = 0
            num_faces_sum  = 0
            t              = time.time()

    G.graph['vis_faces']   = vis_faces
    G.graph['covis_faces'] = covis_faces
    G.graph['area_faces']  = mesh.area_faces


@parame.configurable
def update_edge_visibility(G, mesh, *, force=False, seen=None,
                           load_cache: cfg.param = True,
                           save_cache: cfg.param = True,
                           cache_path: cfg.param = 'runs/cache'):
    "Caching wrapper for _update_edge_visibility"

    cache_file = path.join(cache_path, f'edge_vis_{graph_md5(G)}_{_sensor_directions.md5()}_{mesh.md5()}.npz')

    if load_cache and path.exists(cache_file):
        log.info('loading edge visibility from cache at %s', cache_file)
        try:
            d = np.load(cache_file, allow_pickle=True)
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
        log.info('saving edge visibility to %s', cache_file)
        edges = np.array(G.edges, dtype=object)
        vis_faces_edges  = np.array([G.edges[uv]['vis_faces']  for uv in edges])
        dist_faces_edges = np.array([G.edges[uv]['dist_faces'] for uv in edges])
        makedirs(path.dirname(cache_file), exist_ok=True)
        d = dict(edges=edges,
                 vis_faces_edges=vis_faces_edges,
                 dist_faces_edges=dist_faces_edges,
                 vis_faces=G.graph['vis_faces'],
                 covis_faces=G.graph['covis_faces'])
        np.savez_compressed(cache_file, **d)


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
