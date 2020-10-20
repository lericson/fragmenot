#!./env/bin/python3

import sys
from os import path
from glob import glob

import yaml
import trimesh
import numpy as np
import networkx as nx
from trimesh import grouping

import parame
from utils import ndarrays2graph


cfg = parame.Module('precache')

#: Used for our crappy animation maker
_frame_num = 0


def check(x, y):
    assert 10 < len(x) < 5000, f'10 < (len(x) := {len(x)}) < 5000'
    assert x.shape == y.shape, f'(x.shape := {x.shape}) == (y.shape := {y.shape})'
    assert 0.0000 < x[ 0] <= 0.050, f'0.0000 < (x[-1] := {x[-1]:.4f}) <= 0.050'
    assert 0.9999 < x[-1] <= 1.000, f'0.9999 < (x[-1] := {x[-1]:.4f}) <= 1.000'
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
    if not boundary:
        return 0.0
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
def perimeter(mesh, faces, *,
              visualize:      cfg.param = False,
              make_animation: cfg.param = False):
    edges = mesh.edges_sorted.reshape((-1, 6))[faces].reshape((-1, 2))

    if len(edges) < 2:
        return 0.0

    hashes = grouping.hashable_rows(edges)
    sort   = np.argsort(hashes)
    hashes = hashes[sort]
    unique = hashes[1:] != hashes[:-1]
    unique = np.r_[unique, True]
    unique[1:] &= unique[:-1]

    peri_verts = mesh.vertices[edges[sort][unique]]
    peri_len   = np.sum(np.linalg.norm(np.diff(peri_verts, axis=1), axis=-1))

    if visualize or make_animation:
        import pyglet
        from trimesh.path.exchange.misc import edges_to_path
        from trimesh.path.exchange.load import _create_path
        import gui
        path = _create_path(**edges_to_path(edges[sort][unique], mesh.vertices))
        scene = path.scene()
        scene.add_geometry(mesh)
        scene.set_camera(fov=[90, 90])
        mesh.visual.face_colors[:,     :] = (255*np.r_[0.5, 0.5, 0.5, 1.0]).astype(np.uint8)
        mesh.visual.face_colors[faces, :] = (255*np.r_[0.9, 0.2, 0.2, 0.4]).astype(np.uint8)
        path.colors                       = [(255*np.r_[0.2, 0.9, 0.2, 1.0]).astype(np.uint8)]*len(path.entities)
        viewer = gui.make_viewer(scene)
        area = mesh.area_faces[faces].sum()
        viewer.show_message(f'Perimeter (green): {peri_len:.5g} m\n'
                            f'Area (red): {area:.5g} sqm\n'
                            f'IPR: {area*peri_len**2/(1e-5+area**2):.5g}',
                            key='stats', duration=np.inf)
        if make_animation:
            for b in [False, False, False]:
                pyglet.clock.tick()
                viewer.switch_to()
                viewer.dispatch_events()
                viewer.dispatch_event('on_draw')
                viewer.flip()
            global _frame_num
            viewer.save_image(f'step{_frame_num:05d}.png')
            _frame_num += 1
            viewer.close()
        else:
            gui.run()
            viewer.close()

    return peri_len


def center(mesh, faces):
    return np.mean(mesh.triangles_center[faces]) if len(faces) else 0.0


def area(mesh, faces):
    return np.sum(mesh.area_faces[faces])


@parame.configurable
def precompute_cache(pathname, *,
                     skip_check:     cfg.param = False):

    with open(path.join(pathname, 'environ.yaml')) as f:
        env = yaml.safe_load(f)

    parame._environ  = env['environ']
    parame._file_cfg = env['parame']
    p = parame.cfg['profile']
    parame.set_profile(profile=p)

    mesh = trimesh.Trimesh(**np.load(path.join(pathname, 'mesh.npz')))

    cache_filename = '.plot_cache_v2.npz'
    cache_pathname = path.join(pathname, cache_filename)

    S = sorted(glob(path.join(pathname, 'state?????.npz')))
    assert 10 < len(S) < 5000, f'10 < (len(S) := {len(S)}) < 5000'

    S    = [dict(np.load(fn)) for fn in S]
    x    = np.array([s['completion'] for s in S])
    y    = np.array([s['distance']   for s in S])
    seen = np.array([s['seen_faces'] for s in S])

    if not skip_check:
        check(x, y)

    #roadmap = ndarrays2graph(**np.load(path.join(pathname, 'roadmap.npz')))
    #vis_faces = roadmap.graph['vis_faces']

    # NOTE I reconstructed roadmap.npz's "vis_faces" after-the-fact for some
    # runs using the edge_vis_*.npz cache file. These do not include faces
    # visible when traversing to and from the start state. As a heuristic, add
    # faces visible at any point.
    vis_faces = np.any(seen, axis=0)

    #mesh.update_faces(vis_faces)
    #for u,v,dd_uv in roadmap.edges.data():
    #    dd_uv['vis_faces'] = dd_uv['vis_faces'][vis_faces]

    holes = np.full((seen.shape[0], 1, 3), np.nan)

    #G = nx.Graph([(u, v, {'i': i}) for i, (u, v) in enumerate(mesh.face_adjacency)])

    for i, seen_i in enumerate(seen):

        #seen_indices_i, = np.nonzero(seen_i | ~vis_faces)
        #unseen_comps = [list(comp) for comp, bdry in components(G, outside=set(seen_indices_i))]

        unseen_face_ids,  = np.nonzero(vis_faces & ~seen_i)
        unseen_comps  = [unseen_face_ids] if 0 < len(unseen_face_ids) else []

        bdry_len_i  = [perimeter(mesh, comp) for comp in unseen_comps]
        hole_pos_i  = [   center(mesh, comp) for comp in unseen_comps]
        hole_size_i = [     area(mesh, comp) for comp in unseen_comps]

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
    failed = False

    for i, pathname in enumerate(pathnames):
        print(f'{pathname}: precaching plot data')
        try:
            precompute_cache(pathname)
        except Exception as e:
            print(f'{pathname}: failed: {type(e).__name__}: {e}')
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
