#!/usr/bin/env python3

import sys
from glob import glob
from os import path
from collections import namedtuple

import yaml
import numpy as np
from matplotlib import pyplot as plt

import parame


Desc = namedtuple('Desc', 'run d x y label')

runs = sys.argv[1:]
Ds   = {}


def load(run):

    with open(path.join(run, 'environ.yaml')) as f:
        env = yaml.safe_load(f)

    parame._environ  = env['environ']
    parame._file_cfg = env['parame']
    #d = env['parame']['prevision']['max_distance']
    #p = env['parame']['parame']['profile']
    p = parame.cfg['profile']
    parame.set_profile(profile=p)

    d = parame.Module('prevision')['max_distance']

    print(f'p={p}', end=' ')
    print(f'd={d:.2f}', end=' ')

    cache_filename = path.join(run, '.traj_cache.yaml')

    if not path.exists(cache_filename):
        S = [dict(np.load(fn)) for fn in sorted(glob(path.join(run, 'state?????.npz')))]
        #x = np.array([0.0] + [s['completion'] for s in S] + [1.0])
        #y = np.array([0.0] + [s['distance']   for s in S] + [S[-1]['distance']+1e-8])
        x = np.array([s['completion'] for s in S])
        y = np.array([s['distance']   for s in S])
        print('recalculated', end=' ')

    else:
        with open(cache_filename) as f:
            dd = yaml.safe_load(f)
        x, y = np.array(dd['x']), np.array(dd['y'])
        print('cache', end=' ')

    assert 10 < len(x) < 2000, f'10 < (len(x) := {len(x)}) < 2000'
    assert x.shape == y.shape, f'(x.shape := {x.shape}) == (y.shape := {y.shape})'
    assert 0.000 < x[ 0] < 0.050, f'0.000 < (x[-1] := {x[-1]:.4f}) < 1.000'
    assert 0.998 < x[-1] < 1.000, f'0.998 < (x[-1] := {x[-1]:.4f}) < 1.000'
    assert np.all(np.diff(x[:-1]) >= 0)
    assert np.all(np.diff(y[:-1]) >= 0)

    with open(cache_filename, 'w') as f:
        yaml.dump(dict(x=x.tolist(), y=y.tolist()), f)

    print(f'n={x.shape[0]}', end=' ')
    print()

    return d, x, y


skip_failed = True


for i, run in enumerate(runs):
    print(f'loading {i+1}/{len(runs)}: {run} ', end='')
    try:
        d, x, y = load(run)
    except Exception as e:
        if not skip_failed:
            raise
        else:
            print(f'failed: {e}')
            continue
    desc = Desc(run, d, x, y, f'$d={d:.2f}$')
    Ds.setdefault(d, []).append(desc)

from scipy.interpolate import interp1d

if False: # show histogram
    fin_dists = sorted([ds.y[-1] for d in Ds for ds in Ds[d]])
    h, bins = np.histogram(fin_dists, bins=30)
    for i, d in enumerate(sorted(Ds)):
        plt.hist([ds.y[-1] for ds in Ds[d]], bins=bins, alpha=0.5, color=f'C{i}', label=Ds[d][0].label)
    plt.legend()
    plt.tight_layout()
    plt.show()

keys = sorted(Ds)

if False: # plot histograms
    for completion in (.5, .6, .7, .75, .8, .85, .9, .95, .96, .97, .975, .98, .985, .9875, .99, .998):
        Ys = [[interp1d(ds.x, ds.y)(completion) for ds in Ds[d]] for d in keys]
        plt.bar(np.arange(len(Ds)), [np.mean(ys) for ys in Ys], tick_label=keys, yerr=[np.std(ys) for ys in Ys])
        plt.title(f'{completion*100:.1f}% completion by distance travelled')
        plt.xlabel('Prevision distance $d / m$')
        plt.ylabel('Distance travelled $s / m$')
        plt.savefig(f'completion{completion*1000:.0f}.png')
        plt.show()

        #Ys = [[ds.y[-1] for ds in Ds[d]] for d in keys]
        #plt.bar(np.arange(len(Ds)), [np.mean(ys) for ys in Ys], tick_label=keys, yerr=[np.std(ys) for ys in Ys])
        #plt.title('100% completion by distance travelled')
        #plt.xlabel('Prevision distance $d / m$')
        #plt.ylabel('Distance travelled $s / m$')
        #plt.show()

X = np.linspace(0.050, 0.998, 1000)
for i, d in enumerate(sorted(Ds)):
    F       = [interp1d(ds.x, ds.y, fill_value=(0.0, ds.y.max())) for ds in Ds[d]]
    F_X     = [f(X) for f in F]
    mu_X    = np.mean(F_X, axis=0)
    sigma_X = np.std(F_X, axis=0)
    if True:
        #plt.fill_between(X, mu_X - sigma_X, mu_X + sigma_X, color=f'C{i}', alpha=0.4)
        plt.plot(X, mu_X, color=f'C{i}', alpha=1.0, label=f'$d={d:.2f}, N={len(Ds[d])}$')
    else:
        plt.polar(2*np.pi*X, mu_X, label=f'$d={d:.2f}, N={len(Ds[d])}$')
    #for j, ds in enumerate(Ds[d]):
    #    label = ds.label if j == 0 else None
    #    #plt.plot(FX[j], X, label=label, color=f'C{i}')
    #    plt.plot(ds.x, ds.y, label=label, color=f'C{i}')

plt.title(r'Completion plot with $\pm 1\sigma$ bound')
plt.xlabel('% complete')
plt.ylabel('Distance travelled [m]')
plt.legend()
plt.tight_layout()
plt.show()
