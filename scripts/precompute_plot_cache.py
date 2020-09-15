#!/usr/bin/env python3

import sys
from os import path
from glob import glob

import yaml
import trimesh
import numpy as np
import networkx as nx

import parame
from utils import ndarrays2graph


cfg = parame.Module('precache')


def check(x, y):
    assert 10 < len(x) < 5000, f'10 < (len(x) := {len(x)}) < 5000'
    assert x.shape == y.shape, f'(x.shape := {x.shape}) == (y.shape := {y.shape})'
    assert 0.0000 < x[ 0] <= 0.050, f'0.000 < (x[-1] := {x[-1]:.4f}) <= 0.050'
    assert 0.9999 < x[-1] <= 1.000, f'0.998 < (x[-1] := {x[-1]:.4f}) <= 1.000'
    assert np.all(np.diff(x[:-1]) >= 0)
    assert np.all(np.diff(y[:-1]) >= 0)


def component(adj, source, *, outside):
    "Allows giving an outside set, and reusing it"
    nextlevel = {source} - outside
    component = set()
    boundary = set()
    visited  = set(outside)
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        visited.update(thislevel)
        for v in thislevel:
            component.add(v)
            adj_v = set(adj[v])
            if adj_v & outside:
                boundary.update({(v, v_) for v_ in adj_v & outside})
            nextlevel.update(adj_v - visited)
    return component, boundary


def components(G, *, outside=None):
    outside = set(outside if outside is not None else ())
    for v in G:
        if v not in outside:
            comp, bdry = component(G._adj, v, outside=outside)
            outside.update(comp)
            if comp:
                yield comp, bdry


def boundary_length(G, mesh, boundary):
    # Edges as row indexes of mesh.face_adjacency. Shape: N.
    face_edge_inds = np.array([G.edges[u,v]['i'] for u, v in boundary])
    # Edges as vertex indexes i, j of mesh.vertices indices. Shape: Nx2.
    vert_edge_inds = mesh.face_adjacency_edges[face_edge_inds]
    # Coordinates of vertices for each edge. Shape: Nx2x3.
    edge_verts = mesh.vertices[vert_edge_inds]
    # Direction of each edge in space. Shape: Nx3.
    edge_dirs = np.diff(edge_verts, axis=1)[:, 0, :]
    # The norm of the directions. Shape: N.
    edge_norms = np.linalg.norm(edge_dirs, axis=1)
    # Finally, the sum length of the boundary.
    return np.sum(edge_norms)


@parame.configurable
def precompute_cache(pathname, *,
                     skip_check: cfg.param = False):

    with open(path.join(pathname, 'environ.yaml')) as f:
        env = yaml.safe_load(f)

    parame._environ  = env['environ']
    parame._file_cfg = env['parame']
    p = parame.cfg['profile']
    parame.set_profile(profile=p)

    mesh = trimesh.Trimesh(**np.load(path.join(pathname, 'mesh.npz')))

    cache_filename = '.plot_cache.npz'
    cache_pathname = path.join(pathname, cache_filename)

    S = sorted(glob(path.join(pathname, 'state?????.npz')))
    assert 10 < len(S) < 5000, f'10 < (len(S) := {len(S)}) < 5000'

    S    = [dict(np.load(fn)) for fn in S]
    x    = np.array([s['completion'] for s in S])
    y    = np.array([s['distance']   for s in S])
    seen = np.array([s['seen_faces'] for s in S])

    if not skip_check:
        check(x, y)

    roadmap = ndarrays2graph(**np.load(path.join(pathname, 'roadmap.npz')))
    vis_faces = roadmap.graph['vis_faces']

    # NOTE I reconstructed roadmap.npz's "vis_faces" after-the-fact for some
    # runs using the edge_vis_*.npz cache file. These do not include faces
    # visible when traversing to and from the start state. As a heuristic, add
    # faces visible at any point.
    vis_faces |= np.any(seen, axis=0)

    holes = np.full((seen.shape[0], 200, 3), np.nan)

    G = nx.Graph([(u, v, {'i': i}) for i, (u, v) in enumerate(mesh.face_adjacency)])

    for i, seen_i in enumerate(seen):

        seen_indices_i, = np.nonzero(seen_i | ~vis_faces)

        unseen_comps = [(list(comp), bdry) for comp, bdry in components(G, outside=set(seen_indices_i))]

        hole_size_i = [np.sum(mesh.area_faces[comp])        for comp, bdry in unseen_comps]
        hole_pos_i  = [np.mean(mesh.triangles_center[comp]) for comp, bdry in unseen_comps]
        bdry_len_i  = [boundary_length(G, mesh, bdry)       for comp, bdry in unseen_comps]

        assert np.isclose(np.sum(hole_size_i), np.sum(mesh.area_faces[~seen_i & vis_faces])), \
               'area[ holes ] == area[ unseen visible faces ]'

        if holes.shape[1] < len(unseen_comps):
            holes_new = np.empty((holes.shape[0], 2*holes.shape[1], holes.shape[2]), dtype=holes.dtype)
            holes_new[:, :holes.shape[1], :] = holes
            holes_new[:, holes.shape[1]:, :] = np.nan
            holes     = holes_new

        holes[i, :len(hole_size_i), 0] = hole_size_i
        holes[i, :len(hole_pos_i),  1] = hole_pos_i
        holes[i, :len(bdry_len_i),  2] = bdry_len_i

    np.savez(cache_pathname, x=x, y=y, seen=seen, holes=holes)


def main(*, pathnames=sys.argv[1:]):
    for i, pathname in enumerate(pathnames):
        precompute_cache(pathname)


if __name__ == '__main__':
    main()
