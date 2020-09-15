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
from matplotlib import pyplot as plt
from nptyping import NDArray

import parame


cfg = parame.Module('plot')

pct_unexplored_log10 = cfg.get('pct_unexplored_log10', 3)
pct_complete = 1e-2*(100.0 - 10.0**(-pct_unexplored_log10))


@dataclass
class Desc():
    pathname:   str
    label:      str
    d:          float
    x:          NDArray[(Any,), np.float]
    y:          NDArray[(Any,), np.float]
    seen:       NDArray[(Any, Any), bool]
    hole_sizes: list
    hole_poses: list
    bdry_lens:  list
    mesh:       trimesh.Trimesh

    @property
    def holiness(self):
        return [np.sum((lens_i)[:]) for lens_i in self.bdry_lens]
        #return [len(sizes_i) - 1 for sizes_i in self.hole_sizes]
        #return [np.sum(sizes_i) for sizes_i in self.hole_sizes]
        #return [np.sum(self.mesh.area_faces[~seen_i]) for seen_i in self.seen]


def check(x, y):
    assert 10 < len(x) < 5000, f'10 < (len(x) := {len(x)}) < 5000'
    assert x.shape == y.shape, f'(x.shape := {x.shape}) == (y.shape := {y.shape})'
    assert 0.0000 < x[ 0] <= 0.050, f'{0.0:.5g} < (x[-1] := {x[-1]:.5g}) <= {0.05:.5g}'
    assert pct_complete < x[-1] <= 1.000, f'(pct_complete := {pct_complete:.5g}) < (x[-1] := {x[-1]:.5f}) <= {1.0:.5g}'
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

    print(f'p={p}', end=' ', flush=True)
    print(f'd={d:.2f}', end=' ', flush=True)

    cache_filename = '.plot_cache.npz'
    cache_pathname = path.join(pathname, cache_filename)

    if not path.exists(cache_pathname):
        raise RuntimeError('you need to precompute the plot cache')

    dd    = np.load(cache_pathname)
    x     = np.array(dd['x'])
    y     = np.array(dd['y'])
    seen  = np.array(dd['seen'])
    hole_sizes = [hole_size_i[~np.isnan(hole_size_i)] for hole_size_i in dd['holes'][:, :, 0]]
    hole_poses = [ hole_pos_i[~np.isnan(hole_pos_i)]  for  hole_pos_i in dd['holes'][:, :, 1]]
    bdry_lens  = [ bdry_len_i[~np.isnan(bdry_len_i)]  for  bdry_len_i in dd['holes'][:, :, 2]]

    print(f'n={len(x)}', end=' ', flush=True)

    check(x, y)

    # Make sure we start at x = 0.
    if y[0] == 0.0:
        x[0] = 0.0
    elif x[0] != 0.0:
        x = np.cat(([0.0], x))
        y = np.cat(([0.0], y))

    print()

    desc = Desc(pathname=pathname, label=f'$d={d:.2f}$', d=d, x=x, y=y,
                seen=seen, mesh=mesh, hole_sizes=hole_sizes,
                hole_poses=hole_poses, bdry_lens=bdry_lens)

    return desc


from bisect import bisect_left
def lerp(X, Y, x):
    i = bisect_left(X, x)
    x0, x1 = X[i-1], X[i]
    y0, y1 = Y[i-1], Y[i]
    t = (x - x0)/(x1 - x0)
    y = (1 - t)*y0 + t*y1
    return y

assert lerp([0, 1], [10, 20], 0.0) == 10.0
assert lerp([0, 1], [10, 20], 0.5) == 15.0
assert lerp([0, 1], [10, 20], 1.0) == 20.0


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
    for i, d in enumerate(Ds):
        # Lerp
        XYs_in  = [(ds.y, ds.holiness) for ds in Ds[d]]
        x_max   = sorted(X_in[-1] for X_in, Y_in in XYs_in)[-1]
        h       = x_max/1000
        X_lerp  = h*np.arange(0, 1001)
        Y_lerp  = [[lerp(X_in, Y_in, x_lerp) for X_in, Y_in in XYs_in if X_in[0] <= x_lerp <= X_in[-1]] for x_lerp in X_lerp]

        mu_Y    = np.array([np.mean(Y_lerp[i]) for i in range(len(X_lerp))])
        sigma_Y = np.array([np.std(Y_lerp[i])  for i in range(len(X_lerp))])

        dstr    = f'{d:.2f}' if d < 50.0 else '\infty'
        label   = f'$d={dstr}, N={len(Ds[d])}$'

        ax = plt.gca()
        ax.fill_between(X_lerp, mu_Y - sigma_Y, mu_Y + sigma_Y, color=f'C{i}', alpha=0.1)
        ax.plot(X_lerp, mu_Y, color=f'C{i}', alpha=1.0, label=label)

    plt.title(r'Holiness plot with $\pm 1\sigma$ bound')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #sys.exit()

    if False: # show histogram
        fin_dists = sorted([ds.y[-1] for d in Ds for ds in Ds[d]])
        h, bins = np.histogram(fin_dists, bins=30)
        for i, d in enumerate(Ds):
            plt.hist([ds.y[-1] for ds in Ds[d]], bins=bins, alpha=0.5, color=f'C{i}', label=Ds[d][0].label)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if False: # plot histograms
        for completion in (.5, .6, .7, .75, .8, .85, .9, .95, .96, .97, .975, .98, .985, .9875, .99, .998):
            Ys = [[interp1d(ds.x, ds.y)(completion) for ds in Ds[d]] for d in Ds]
            plt.bar(np.arange(len(Ds)), [np.mean(ys) for ys in Ys], tick_label=list(Ds), yerr=[np.std(ys) for ys in Ys])
            plt.title(f'{completion*100:.1f}% completion by distance travelled')
            plt.xlabel('Prevision distance $d / m$')
            plt.ylabel('Distance travelled $s / m$')
            plt.savefig(f'completion{completion*1000:.0f}.png')
            plt.show()

            #Ys = [[ds.y[-1] for ds in Ds[d]] for d in Ds]
            #plt.bar(np.arange(len(Ds)), [np.mean(ys) for ys in Ys], tick_label=list(Ds), yerr=[np.std(ys) for ys in Ys])
            #plt.title('100% completion by distance travelled')
            #plt.xlabel('Prevision distance $d / m$')
            #plt.ylabel('Distance travelled $s / m$')
            #plt.show()

    from matplotlib import ticker, scale
    for i, d in enumerate(Ds):
        X        = np.linspace(0.0, pct_complete, 1000)
        F        = [interp1d(ds.x, ds.y, fill_value=(0.0, ds.y.max())) for ds in Ds[d]]
        FX       = [f(X) for f in F]
        mu_FX    = np.mean(FX, axis=0)
        sigma_FX = np.std(FX, axis=0)
        dstr     = f'{d:.2f}' if d < 50.0 else '\infty'
        label    = f'$d={dstr}, N={len(Ds[d])}$'
        if style == 'line':
            ax = plt.gca()
            X = 100*(1 - X)
            ax.fill_between(X, mu_FX - sigma_FX, mu_FX + sigma_FX, color=f'C{i}', alpha=0.1)
            ax.plot(X, mu_FX, color=f'C{i}', alpha=1.0, label=label)
            ax.plot([0, 100], [mu_FX[-1], mu_FX[-1]], '--', color=f'C{i}', linewidth=1, alpha=0.6)
            ax.set_xscale('log', basex=10)
            ax.set_xlim([X.max(), X.min()])
            #ax.xaxis._scale = scale.LogScale(ax.xaxis, basex=1.2)
            #ax.xaxis.set_major_locator(ticker.LogLocator())
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3g}%'))
            plt.xlabel('Unexplored surface area [%]')
            plt.ylabel('Distance travelled [m]')
        elif style == 'polar':
            ax = plt.gca(polar=True)
            ax.plot(2*np.pi*X, mu_FX, label=label)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/2/np.pi*100:.0f}%'))
        else:
            raise ValueError(style)
        #for j, ds in enumerate(Ds[d]):
        #    label = ds.label if j == 0 else None
        #    #plt.plot(FX[j], X, label=label, color=f'C{i}')
        #    plt.plot(ds.x, ds.y, label=label, color=f'C{i}')

    plt.title(r'Completion plot with $\pm 1\sigma$ bound')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
