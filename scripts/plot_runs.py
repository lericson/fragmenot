#!/usr/bin/env python3

import sys
from os import path
from glob import glob
from typing import Any, Optional
from dataclasses import dataclass
from collections import namedtuple

import yaml
import trimesh
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt, ticker, scale
from nptyping import NDArray

import parame
from utils import ndarrays2graph


cfg = parame.Module('plot')

pct_unexplored_log10 = cfg.get('pct_unexplored_log10', 3)
thr_complete = 1e-2*(100.0 - 10.0**(-pct_unexplored_log10))


@dataclass
class Desc():
    pathname:   str
    label:      str
    d:          float
    complete:   NDArray[(Any,), np.float]
    distance:   NDArray[(Any,), np.float]
    seen:       NDArray[(Any, Any), bool]
    hole_sizes: list
    hole_poses: list
    bdry_lens:  list
    mesh:       trimesh.Trimesh
    vis_faces:  NDArray[(Any,), bool]

    @property
    def boundary_length(self):
        return [np.sum((lens_i)[:]) for lens_i in self.bdry_lens]
        #return [np.linalg.norm(np.var(poses_i, axis=0)) for poses_i in self.hole_poses]
        #return [len(sizes_i) - 1 for sizes_i in self.hole_sizes]
        #return [np.sum(sizes_i) for sizes_i in self.hole_sizes]
        #return [np.sum(self.mesh.area_faces[~seen_i]) for seen_i in self.seen]

    @property
    def seen_area(self):
        return [np.sum(self.mesh.area_faces[seen_i]) for seen_i in self.seen]

    @property
    def vis_area(self):
        return np.sum(self.mesh.area_faces[self.vis_faces])


def check(x, y):
    assert 10 < len(x) < 5000, f'10 < (len(x) := {len(x)}) < 5000'
    assert x.shape == y.shape, f'(x.shape := {x.shape}) == (y.shape := {y.shape})'
    assert 0.0000 < x[ 0] <= 0.050, f'{0.0:.5g} < (x[-1] := {x[-1]:.5g}) <= {0.05:.5g}'
    assert thr_complete < x[-1] <= 1.000, f'(thr_complete := {thr_complete:.5g}) < (x[-1] := {x[-1]:.5f}) <= {1.0:.5g}'
    assert np.all(np.diff(x[:-1]) >= 0)
    assert np.all(np.diff(y[:-1]) >= 0)


def bfs(adj, source, *, seen=None):
    "BFS but optionally with an initial *seen*"
    if seen is None:
        seen = set()
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


def load(pathname):

    with open(path.join(pathname, 'environ.yaml')) as f:
        env = yaml.safe_load(f)

    parame._environ  = env['environ']
    parame._file_cfg = env['parame']
    #d = env['parame']['prevision']['max_distance']
    #p = env['parame']['parame']['profile']
    p = parame.cfg['profile']
    parame.set_profile(profile=p)

    d = parame.Module('prevision')['max_distance']

    mesh = trimesh.Trimesh(**np.load(path.join(pathname, 'mesh.npz')))
    roadmap = ndarrays2graph(**np.load(path.join(pathname, 'roadmap.npz')))

    print(f'p={p}', end=' ', flush=True)
    print(f'd={d:.2f}', end=' ', flush=True)

    cache_filename = '.plot_cache.npz'
    cache_pathname = path.join(pathname, cache_filename)

    if not path.exists(cache_pathname):
        raise RuntimeError('you need to precompute the plot cache')

    dd         = np.load(cache_pathname)
    complete   = np.array(dd['x'])
    distance   = np.array(dd['y'])
    seen       = np.array(dd['seen'])
    hole_sizes = [hole_size_i[~np.isnan(hole_size_i)] for hole_size_i in dd['holes'][:, :, 0]]
    hole_poses = [ hole_pos_i[~np.isnan(hole_pos_i)]  for  hole_pos_i in dd['holes'][:, :, 1]]
    bdry_lens  = [ bdry_len_i[~np.isnan(bdry_len_i)]  for  bdry_len_i in dd['holes'][:, :, 2]]
    vis_faces  = roadmap.graph['vis_faces']

    # NOTE I reconstructed roadmap.npz's "vis_faces" after-the-fact for some
    # runs using the edge_vis_*.npz cache file. These do not include faces
    # visible when traversing to and from the start state. As a heuristic, add
    # faces visible at any point.
    vis_faces |= np.any(seen, axis=0)
    # Conversely, we pretend like faces not in vis_faces were never seen.
    seen &= vis_faces

    print(f'n={len(complete)}', end=' ', flush=True)

    check(complete, distance)

    # Make sure we start at dist = 0.
    if distance[0] == 0.0:
        complete[0] = 0.0
    elif complete[0] != 0.0:
        complete = np.cat(([0.0], complete))
        distance = np.cat(([0.0], distance))

    print()

    desc = Desc(pathname=pathname, label=f'$d={d:.2f}$', d=d, complete=complete,
                distance=distance, seen=seen, mesh=mesh, hole_sizes=hole_sizes,
                hole_poses=hole_poses, bdry_lens=bdry_lens, vis_faces=vis_faces)

    return desc


from bisect import bisect_left
def lerp(X, Y, x):
    "Linear interpolation of y=f(x) in sorted domain x"
    i = bisect_left(X, x)
    x0, x1 = X[i-1], X[i]
    y0, y1 = Y[i-1], Y[i]
    t = (x - x0)/(x1 - x0)
    y = (1 - t)*y0 + t*y1
    return y


# wHY doN't YOu hAvE UNit TeStS
assert lerp([0, 1], [10, 20], 0.0) == 10.0
assert lerp([0, 1], [10, 20], 0.5) == 15.0
assert lerp([0, 1], [10, 20], 1.0) == 20.0


def lerp_defined(Xs, Ys, x):
    """Consider each pair Xs[i], Ys[i] a piecewise linear function. This
    returns a list with their value at x _if it is defined_, i.e. x is in the
    i'th pair's domain."""
    return [lerp(X, Y, x) for X, Y in zip(Xs, Ys) if X[0] <= x <= X[-1]]


def lerp_stats(Xs, Ys, X_lerp):
    """Lerp like lerp_defined but over a set of values X_lerp, then return
    means and standard deviations for each interpolation point."""
    Y_lerp  = [lerp_defined(Xs, Ys, x_i) for x_i in X_lerp]
    mu_Y    = np.array([np.mean(Y_lerp[i], axis=0) for i in range(len(X_lerp))])
    sigma_Y = np.array([np.std(Y_lerp[i], axis=0)  for i in range(len(X_lerp))])
    return mu_Y, sigma_Y


def label(d, Ds, **kwargs):
    dstr = f'{d:.2f}' if d < 50.0 else '\infty'
    return f'$d={dstr}, N={len(Ds[d])}$'


def plot_bounds(X, mean, std, *, k=1, label=None, color=None):
    ax = plt.gca()
    ax.fill_between(X, mean - k*std, mean + k*std, alpha=0.1, color=color)
    ax.plot(        X, mean,                       alpha=1.0, color=color, label=label)


@parame.configurable
def main(*, pathnames=sys.argv[1:],
         style:       cfg.param = 'line',
         skip_failed: cfg.param = True):

    Ds = {}

    for i, pathname in enumerate(pathnames):
        print(f'loading {i+1:3d} / {len(pathnames)}: {pathname} ', end='')
        try:
            desc = load(pathname)
        except Exception as e:
            if not skip_failed:
                raise
            else:
                print(f'failed: {e}')
                continue
        Ds.setdefault(desc.d, []).append(desc)

    if not Ds:
        return

    Ds = {d: Ds[d] for d in sorted(Ds)}

    from scipy.interpolate import interp1d

    #bins = 10
    #_, bin_edges = np.histogram([h for holes_i in desc.holes for h in holes_i], bins=bins)

    print('lerp me like one of your french girls')
    # The domain X for each run is used to compute the lerp domain X_lerp

    for i, d in enumerate(Ds):
        Xs_in         = [ds.distance for ds in Ds[d]]
        x_max         = sorted(X_in[-1] for X_in in Xs_in)[-1]
        h             = x_max/1000
        X_lerp        = h*np.arange(0, 1001)
        Ys_in         = [ds.boundary_length for ds in Ds[d]]
        mu_Y, sigma_Y = lerp_stats(Xs_in, Ys_in, X_lerp)
        plot_bounds(X_lerp, mu_Y, sigma_Y, color=f'C{i}', label=label(**locals()))

    plt.xlabel('Distance travelled [m]')
    plt.ylabel('Frontier length [m]')
    plt.title(r'Frontier length with $\pm 1\sigma$ bound')
    plt.legend()
    plt.tight_layout()
    plt.show()

    for i, d in enumerate(Ds):
        Xs_in         = [ds.distance for ds in Ds[d]]
        x_max         = sorted(X_in[-1] for X_in in Xs_in)[-1]
        h             = x_max/1000
        X_lerp        = h*np.arange(0, 1001)
        Ys_in         = [100*(1 + (1 - thr_complete) - ds.complete) for ds in Ds[d]]
        mu_Y, sigma_Y = lerp_stats(Xs_in, Ys_in, X_lerp)
        plot_bounds(X_lerp, mu_Y, sigma_Y, color=f'C{i}', label=label(**locals()))
        #plt.plot([0, 100], [mu_Y[-1], mu_Y[-1]], '--', color=f'C{i}', linewidth=1, alpha=0.6)

    ax = plt.gca()
    ax.set_yscale('log', basey=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3g}%'))
    #ax.set_ylim([100.0, 0])

    plt.xlabel('Distance travelled [m]')
    plt.ylabel('Unexplored')
    plt.title(r'Completion plot with $\pm 1\sigma$ bound')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #sys.exit()

    if False: # show histogram
        fin_dists = sorted([ds.distance[-1] for d in Ds for ds in Ds[d]])
        h, bins = np.histogram(fin_dists, bins=30)
        for i, d in enumerate(Ds):
            plt.hist([ds.distance[-1] for ds in Ds[d]], bins=bins, alpha=0.5, color=f'C{i}', label=Ds[d][0].label)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if False: # plot histograms
        for completion in (.5, .6, .7, .75, .8, .85, .9, .95, .96, .97, .975, .98, .985, .9875, .99, .998):
            Ys = [[interp1d(ds.complete, ds.distance)(completion) for ds in Ds[d]] for d in Ds]
            plt.bar(np.arange(len(Ds)), [np.mean(ys) for ys in Ys], tick_label=list(Ds), yerr=[np.std(ys) for ys in Ys])
            plt.title(f'{completion*100:.1f}% completion by distance travelled')
            plt.xlabel('Prevision distance $d / m$')
            plt.ylabel('Distance travelled $s / m$')
            plt.savefig(f'completion{completion*1000:.0f}.png')
            plt.show()

            #Ys = [[ds.distance[-1] for ds in Ds[d]] for d in Ds]
            #plt.bar(np.arange(len(Ds)), [np.mean(ys) for ys in Ys], tick_label=list(Ds), yerr=[np.std(ys) for ys in Ys])
            #plt.title('100% completion by distance travelled')
            #plt.xlabel('Prevision distance $d / m$')
            #plt.ylabel('Distance travelled $s / m$')
            #plt.show()

    for i, d in enumerate(Ds):
        X        = np.linspace(0.0, thr_complete, 1000)
        F        = [interp1d(ds.complete, ds.distance, fill_value=(0.0, ds.distance.max())) for ds in Ds[d]]
        FX       = [f(X) for f in F]
        mu_FX    = np.mean(FX, axis=0)
        sigma_FX = np.std(FX, axis=0)
        if style == 'line':
            ax = plt.gca()
            X = 100*(1 - X)
            ax.fill_between(X, mu_FX - sigma_FX, mu_FX + sigma_FX, color=f'C{i}', alpha=0.1)
            ax.plot(X, mu_FX, color=f'C{i}', alpha=1.0, label=label(**locals()))
            ax.plot([0, 100], [mu_FX[-1], mu_FX[-1]], '--', color=f'C{i}', linewidth=1, alpha=0.6)
            ax.set_xscale('log', basex=10)
            ax.set_xlim([X.max(), X.min()])
            #ax.xaxis._scale = scale.LogScale(ax.xaxis, basex=1.2)
            #ax.xaxis.set_major_locator(ticker.LogLocator())
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3g}%'))
        elif style == 'polar':
            ax = plt.gca(polar=True)
            ax.plot(2*np.pi*X, mu_FX, label=label(**locals()))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/2/np.pi*100:.0f}%'))
        else:
            raise ValueError(style)
        #for j, ds in enumerate(Ds[d]):
        #    label = ds.label if j == 0 else None
        #    #plt.plot(FX[j], X, label=label, color=f'C{i}')
        #    plt.plot(ds.complete, ds.distance, label=label, color=f'C{i}')

    plt.xlabel('Unexplored surface area [%]')
    plt.ylabel('Distance travelled [m]')
    plt.title(r'Completion plot with $\pm 1\sigma$ bound')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
