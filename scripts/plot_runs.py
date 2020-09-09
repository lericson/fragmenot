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


@dataclass
class Desc():
    pathname:   str
    label:      str
    d:          float
    x:          NDArray[(Any,), np.float]
    y:          NDArray[(Any,), np.float]

    seen:       NDArray[(Any, Any), bool]     = None
    holes:      NDArray[(Any,),     np.float] = None
    faces:      NDArray[(Any, 3),   np.int]   = None
    vertices:   NDArray[(Any, 3),   np.float] = None
    mesh:       trimesh.Trimesh               = None


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
        print('recalculated', end=' ')

    else:
        dd    = np.load(cache_pathname)
        x     = np.array(dd['x'])
        y     = np.array(dd['y'])
        seen  = np.array(dd['seen'])
        holes = np.array(dd['holes'])
        check(x, y)
        print('cache', end=' ')

    # Make sure we start at x = 0.
    if y[0] == 0.0:
        x[0] = 0.0
    elif x[0] != 0.0:
        x = np.cat(([0.0], x))
        y = np.cat(([0.0], y))

    print(f'n={x.shape[0]}', end=' ')
    print()

    desc = Desc(pathname=pathname, label=f'$d={d:.2f}$', d=d, x=x, y=y,
                seen=seen, holes=holes, mesh=mesh)

    return desc


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

    Ds = {d: Ds[d] for d in sorted(Ds)}

    from scipy.interpolate import interp1d

    if False: # show histogram
        fin_dists = sorted([ds.y[-1] for d in Ds for ds in Ds[d]])
        h, bins = np.histogram(fin_dists, bins=30)
        for i, d in enumerate(sorted(Ds)):
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
    for i, d in enumerate(sorted(Ds)):
        X        = np.linspace(0.0, 1e-2*(100.0 - 0.001), 1000)
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
