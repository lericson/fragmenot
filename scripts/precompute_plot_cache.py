#!/usr/bin/env python3

import sys
from os import path
from glob import glob

import yaml
import trimesh
import numpy as np
import networkx as nx

import parame


def check(x, y):
    assert 10 < len(x) < 2000, f'10 < (len(x) := {len(x)}) < 2000'
    assert x.shape == y.shape, f'(x.shape := {x.shape}) == (y.shape := {y.shape})'
    assert 0.0000 < x[ 0] <= 0.050, f'0.000 < (x[-1] := {x[-1]:.4f}) <= 0.050'
    assert 0.9999 < x[-1] <= 1.000, f'0.998 < (x[-1] := {x[-1]:.4f}) <= 1.000'
    assert np.all(np.diff(x[:-1]) >= 0)
    assert np.all(np.diff(y[:-1]) >= 0)


def bfs(adj, source, *, seen):
    "Allows giving an initial seen, and reusing it"
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel - seen
        nextlevel = set()
        for v in thislevel:
            yield v
            nextlevel.update(adj[v])
        seen.update(thislevel)


def components(G, *, seen=None):
    seen = seen if seen is not None else set()
    return filter(None, (list(bfs(G.adj, v, seen=seen)) for v in G if v not in seen))


def find_holes(pathname):

    with open(path.join(pathname, 'environ.yaml')) as f:
        env = yaml.safe_load(f)

    parame._environ  = env['environ']
    parame._file_cfg = env['parame']
    #d = env['parame']['prevision']['max_distance']
    #p = env['parame']['parame']['profile']
    p = parame.cfg['profile']
    parame.set_profile(profile=p)

    mesh = trimesh.Trimesh(**np.load(path.join(pathname, 'mesh.npz')))

    cache_filename = '.plot_cache.npz'
    cache_pathname = path.join(pathname, cache_filename)

    S = sorted(glob(path.join(pathname, 'state?????.npz')))
    assert 10 < len(S) < 2000, f'10 < (len(S) := {len(S)}) < 2000'
    S = [dict(np.load(fn)) for fn in S]
    x    = np.array([s['completion'] for s in S])
    y    = np.array([s['distance']   for s in S])
    seen = np.array([s['seen_faces'] for s in S])

    holes = []
    G = nx.Graph(list(mesh.face_adjacency))
    for i, seen_i in enumerate(seen):
        seen_indices_i, = np.nonzero(seen[i])
        holes.append([np.sum(mesh.area_faces[comp]) for comp in components(G, seen=set(seen_indices_i))])
    holes = np.asarray(holes)

    check(x, y)

    np.savez(cache_pathname, x=x, y=y, seen=seen, holes=holes)


@parame.configurable
def main(*, pathnames=sys.argv[1:]):
    for i, pathname in enumerate(pathnames):
        find_holes(pathname)


if __name__ == '__main__':
    main()
