#!./env/bin/python3

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
from matplotlib import pyplot as plt, ticker, scale, patches, rc
from nptyping import NDArray

import parame
from utils import ndarrays2graph


cfg = parame.Module('plot')

pct_unexplored_log10 = cfg.get('pct_unexplored_log10', 3)
thr_complete = 1e-2*(100.0 - 10.0**(-pct_unexplored_log10))

rc('lines', linewidth=1.0)
rc('font', family='serif', size=10.0)
#rc('pdf', fonttype=42)
rc('text', usetex=True)
rc('figure', figsize=1.2*np.r_[84.0, 32.0]/25)

#colors = 'C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16'.split()
colors = 'C0 C3 C2 C4 C5 C6 C7 C8 C9 C1 C10 C11 C12 C13 C14 C15 C16'.split()


@dataclass
class Desc():
    pathname:   str
    label:      str
    d:          float
    complete:   NDArray[(Any,), np.float]
    distance:   NDArray[(Any,), np.float]
    seen:       NDArray[(Any, Any), bool]
    mesh:       trimesh.Trimesh
    vis_faces:  NDArray[(Any,), bool]
    unseen_perimeters: list
    unseen_areas: list
    env:        dict

    @property
    def isoperimetric_ratio(self):
        L     = np.array(list(map(sum, self.unseen_perimeters)))
        A     = np.array(list(map(sum, self.unseen_areas)))
        # We do IPR like this because AL² / (ε + A²) -> 0 as A -> 0
        return A*L**2 / (1e-9 + A**2)


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

    print(f'p={p}', end=' ', flush=True)
    print(f'd={d:.2f}', end=' ', flush=True)

    cache_filename = '.plot_cache_v2.npz'
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

    # NOTE I reconstructed roadmap.npz's "vis_faces" after-the-fact for some
    # runs using the edge_vis_*.npz cache file. These do not include faces
    # visible when traversing to and from the start state. As a heuristic, add
    # faces visible at any point.
    vis_faces = np.any(seen, axis=0)
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

    desc = Desc(pathname=pathname, label=f'$d={d:.2f}$', d=d, complete=complete,
                distance=distance, seen=seen, mesh=mesh, unseen_areas=hole_sizes,
                unseen_perimeters=bdry_lens, vis_faces=vis_faces, env=env)

    desc.profile = parame.cfg['profile']

    return desc


from bisect import bisect_left
def lerp(X, Y, x):
    "Linear interpolation of y=f(x) in sorted domain x"
    i = bisect_left(X, x)
    if i == 0:
        return Y[0]
    if i == len(X):
        return Y[-1]
    x0, x1 = X[i-1], X[i]
    y0, y1 = Y[i-1], Y[i]
    t = (x - x0)/(x1 - x0)
    y = (1 - t)*y0 + t*y1
    return y


# wHY doN't YOu hAvE UNit TeStS
assert lerp([0, 1], [10, 20], 0.0) == 10.0
assert lerp([0, 1], [10, 20], 0.5) == 15.0
assert lerp([0, 1], [10, 20], 1.0) == 20.0
assert lerp([0, 1], [10, 20], 2.0) == 20.0
assert lerp([0, 1], [10, 20], -1.0) == 10.0


def lerp_defined(Xs, Ys, x):
    """Consider each pair Xs[i], Ys[i] a piecewise linear function. This
    returns a list with their value at x _if it is defined_, i.e. x is in the
    i'th pair's domain."""
    return [lerp(X, Y, x) for X, Y in zip(Xs, Ys) if X[0] <= x <= X[-1]]


def lerp_many(Xs, Ys, x):
    return [lerp(X, Y, x) for X, Y in zip(Xs, Ys)]


def lerp_stats(Xs, Ys, X_lerp, *, only_defined=True):
    """Lerp like lerp_defined but over a set of values X_lerp, then return
    means and standard deviations for each interpolation point."""
    if only_defined:
        Y_lerp = [lerp_defined(Xs, Ys, x_i) for x_i in X_lerp]
    else:
        Y_lerp = [lerp_many(Xs, Ys, x_i) for x_i in X_lerp]
    mu_Y    = np.array([np.mean(Y_lerp[i], axis=0) for i in range(len(X_lerp))])
    sigma_Y = np.array([np.std(Y_lerp[i], axis=0)  for i in range(len(X_lerp))])
    return mu_Y, sigma_Y


def label(k, Group, **kwargs):
    if isinstance(k, str):
        return k
    d = Group[k][0].d
    dstr = f'{d:.2f}' if d < 50.0 else '$\infty$'
    #return r'$d$ =\hphantom{99.99}\llap{' + dstr + r'}, $N$ = ' + str(len(Group[k]))
    return r'$d$ =\hphantom{99.99}\llap{' + dstr + r'}'


def plot_lerp(X, Y, points, *, k=1, interval=0.05, label=None, color=None):
    q = (interval, 0.5, 1 - interval)
    Y_points    = [lerp_many(X, Y, pt) for pt in points]
    Y_quantiles = np.array([np.quantile(Y_pt, q) for Y_pt in Y_points])
    Y_mean      = np.array([np.mean(Y_pt)        for Y_pt in Y_points])
    ax = plt.gca()
    ax.fill_between(points, Y_quantiles[:, 0], Y_quantiles[:, 2], alpha=0.1, color=color)
    ax.plot(        points, Y_quantiles[:, 1],                    alpha=1.0, color=color, label=label)


@parame.configurable
def main(*, args=sys.argv[1:],
         style:       cfg.param = 'line',
         ncol:        cfg.param = 5,
         save_legend: cfg.param = None,
         save_ipr:    cfg.param = None,
         save_compl:  cfg.param = None,
         skip_failed: cfg.param = True):

    Group = {}

    import argparse
    subparser = argparse.ArgumentParser()
    subparser.add_argument('--group')
    group = subparser.add_mutually_exclusive_group()
    group.add_argument('--filter')
    group.add_argument('--d', type=float)
    group.add_argument('--profile', action='append')
    subparser.add_argument('pathnames', nargs='+')

    #args = subparser.parse_intermixed_args()

    arg_sets = [[]]
    for arg in args:
        if arg == '--':
            arg_sets.append([])
        else:
            arg_sets[-1].append(arg)

    namespaces = [subparser.parse_args(args) for args in arg_sets]
    ninputs = sum([len(ns.pathnames) for ns in namespaces])
    inputno = 0

    for i, ns in enumerate(namespaces):
        if ns.filter:
            ns.filter = eval('lambda desc: ' + ns.filter)
        elif ns.d:
            ns.filter = lambda desc: np.isclose(desc.d, ns.d)
        elif ns.profile:
            profiles = set(ns.profile)
            ns.filter = lambda desc: profiles < set(desc.profile)
        else:
            ns.filter = lambda desc: True
        for inputno, pathname in enumerate(ns.pathnames, inputno+1):
            print(f'loading {inputno:3d} / {ninputs}: {pathname} ', end='')
            try:
                desc = load(pathname)
            except Exception as e:
                if not skip_failed:
                    raise
                else:
                    print(f'failed: {e}')
                    continue

            if ns.filter and not ns.filter(desc):
                print('filtered')
                continue
            else:
                print()

            k = ns.group if ns.group else (i, desc.d)
            Group.setdefault(k, []).append(desc)

    assert Group

    Group = {k: Group[k] for k in sorted(Group)}

    print(f'Group keys, sorted:')
    print('\n'.join(str(k) for k in Group))

    from scipy.interpolate import interp1d

    #bins = 10
    #_, bin_edges = np.histogram([h for holes_i in desc.holes for h in holes_i], bins=bins)

    if 0 or save_legend:
        ps = [patches.Patch(color=colors[i], label=label(Group=Group, i=i, k=k))
              for i, k in enumerate(Group)]
        ps = np.array(ps)
        # How we want the grid, i.e., [[0, 1, 2], [3, 4, 5]]
        grid = np.arange(ncol*(len(ps)//ncol + 1)).reshape(-1, ncol)
        # Inverse of how the grid is laid out, i.e., how it is input to MPL.
        invert = grid.transpose().reshape(-1)
        # Apply the inverse mapping to our legend patches
        ps = ps[invert[invert < len(ps)]]
        fig = plt.figure()
        fig.legend(ps, [p.get_label() for p in ps], loc='center', frameon=True, ncol=ncol)
        if save_legend:
            fig.savefig(save_legend)
        else:
            plt.show()
            print('Please save legend.')
            print('Press Enter to continue, ^C to Exit')
            try:
                sys.stdin.detach().read(1)
            except (EOFError, KeyboardInterrupt):
                print('Exiting')
                sys.exit()

    print('LERP me like one of your French girls')

    if 1 or save_ipr:
        fig = plt.figure()

        for i, k in enumerate(Group):
            Xs_in         = [ds.distance for ds in Group[k]]
            x_max         = sorted(X_in[-1] for X_in in Xs_in)[-1]
            h             = x_max/1000
            X_lerp        = h*np.arange(0, 1001)
            Ys_in         = [ds.isoperimetric_ratio for ds in Group[k]]
            mu_Y, sigma_Y = lerp_stats(Xs_in, Ys_in, X_lerp)
            plot_lerp(Xs_in, Ys_in, X_lerp, color=colors[i], label=label(**locals()))

        ax = fig.gca()
        ax.set_yscale('log', basey=10)
        ax.set_xlim([ 0.0, cfg.get('xmax', 1000)])
        ax.set_ylim([20.0, cfg.get('ymax', 6000)])
        loc = ax.yaxis.get_major_locator()
        #loc.set_params(min_n_ticks=4)
        plt.xlabel('Distance Traveled [m]')
        plt.ylabel('IPR')
        plt.grid(True)
        #plt.title('IPR of unseen space (95\\% CI)')
        #plt.legend(ncol=2)
        #plt.tight_layout()
        plt.subplots_adjust(top=0.961, bottom=0.282, left=0.142, right=0.963)
        if save_ipr:
            fig.savefig(save_ipr)
        else:
            plt.show()

    if 1 or save_compl:
        fig = plt.figure()

        for i, k in enumerate(Group):
            Xs_in         = [ds.distance for ds in Group[k]]
            x_max         = sorted(X_in[-1] for X_in in Xs_in)[-1]
            h             = x_max/1000
            X_lerp        = h*np.arange(0, 1001)
            Ys_in         = [100*(1.0 - ds.complete) for ds in Group[k]]
            mu_Y, sigma_Y = lerp_stats(Xs_in, Ys_in, X_lerp)
            plot_lerp(Xs_in, Ys_in, X_lerp, color=colors[i], label=label(**locals()))
            #plt.plot([0, 100], [mu_Y[-1], mu_Y[-1]], '--', color=colors[i], alpha=0.6)

        ax = fig.gca()
        ax.set_yscale('log', basey=10)
        ax.set_xlim([0.0, cfg.get('xmax', 1000)])
        ax.set_ylim([10**-pct_unexplored_log10, 10**2])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3g}\\%'))
        #ax.set_ylim([100.0, 0])
        loc = ax.yaxis.get_major_locator()
        loc.set_params(numticks=pct_unexplored_log10+4)
        loc = ax.yaxis.get_minor_locator()
        loc.set_params(numticks=pct_unexplored_log10+4)
        loc.subs(np.arange(2.0, 10.0, 2))
        #ax.yaxis.set_major_locator(ticker.LogLocator(numticks=4))

        plt.xlabel('Distance Traveled [m]')
        plt.ylabel('Unexplored Area')
        plt.grid(True, which='major')
        #plt.title(r'Completion plot with $\pm 1\sigma$ bound')
        #plt.legend(ncol=4)
        #plt.tight_layout()
        plt.subplots_adjust(top=0.961, bottom=0.282, left=0.177, right=0.963)
        if save_compl:
            fig.savefig(save_compl)
        else:
            plt.show()


    if False: # show histogram
        fin_dists = sorted([ds.distance[-1] for k in Group for ds in Group[k]])
        h, bins = np.histogram(fin_dists, bins=30)
        for i, k in enumerate(Group):
            plt.hist([ds.distance[-1] for ds in Group[k]], bins=bins, alpha=0.5, color=colors[i], label=Group[k][0].label)
        plt.legend()
        plt.tight_layout()
        plt.show()


    if False: # plot histograms
        for completion in (.5, .6, .7, .75, .8, .85, .9, .95, .96, .97, .975, .98, .985, .9875, .99, .998):
            Ys = [[interp1d(ds.complete, ds.distance)(completion) for ds in Group[k]] for k in Group]
            plt.bar(np.arange(len(Group)), [np.mean(ys) for ys in Ys], tick_label=list(Group), yerr=[np.std(ys) for ys in Ys])
            plt.title(f'{completion*100:.1f}\\% completion by distance travelled')
            plt.xlabel('Prevision Distance $d / m$')
            plt.ylabel('Distance Traveled $s / m$')
            plt.savefig(f'completion{completion*1000:.0f}.png')
            plt.show()

            #Ys = [[ds.distance[-1] for ds in Group[k]] for k in Group]
            #plt.bar(np.arange(len(Group)), [np.mean(ys) for ys in Ys], tick_label=list(Group), yerr=[np.std(ys) for ys in Ys])
            #plt.title('100\\% completion by distance travelled')
            #plt.xlabel('Prevision distance $d / m$')
            #plt.ylabel('Distance travelled $s / m$')
            #plt.show()


    if False:
        for i, k in enumerate(Group):
            X        = np.linspace(0.0, thr_complete, 1000)
            F        = [interp1d(ds.complete, ds.distance, fill_value=(0.0, ds.distance.max())) for ds in Group[k]]
            FX       = [f(X) for f in F]
            mu_FX    = np.mean(FX, axis=0)
            sigma_FX = np.std(FX, axis=0)
            if style == 'line':
                ax = plt.gca()
                X = 100*(1 - X)
                ax.fill_between(X, mu_FX - sigma_FX, mu_FX + sigma_FX, color=colors[i], alpha=0.1)
                ax.plot(X, mu_FX, color=colors[i], alpha=1.0, label=label(**locals()))
                ax.plot([0, 100], [mu_FX[-1], mu_FX[-1]], '--', color=colors[i], alpha=0.6)
                print(label(**locals()), f': distance at completion μ={mu_FX[-1]:.5g}, σ={sigma_FX[-1]:.5g}')
                ax.set_xscale('log', basex=10)
                ax.set_xlim([X.max(), X.min()])
                #ax.xaxis._scale = scale.LogScale(ax.xaxis, basex=1.2)
                #ax.xaxis.set_major_locator(ticker.LogLocator())
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.3g}\\%'))
            elif style == 'polar':
                ax = plt.gca(polar=True)
                ax.plot(2*np.pi*X, mu_FX, label=label(**locals()))
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/2/np.pi*100:.0f}\\%'))
            else:
                raise ValueError(style)
            #for j, ds in enumerate(Group[k]):
            #    label = ds.label if j == 0 else None
            #    #plt.plot(FX[j], X, label=label, color=colors[i])
            #    plt.plot(ds.complete, ds.distance, label=label, color=colors[i])

        plt.xlabel('Unexplored surface area [\\%]')
        plt.ylabel('Distance travelled [m]')
        plt.title(r'Completion plot with $\pm 1\sigma$ bound')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
