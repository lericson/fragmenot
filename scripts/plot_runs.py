#!/usr/bin/env python3

import sys
from glob import glob
from os import path
from collections import namedtuple

import yaml
import numpy as np
from matplotlib import pyplot as plt


Desc = namedtuple('Desc', 'run d x y label')

runs = sys.argv[1:]
Ds   = {}

"""
x = np.linspace(0,3)
k, m = 1, 0
y = k*x + m
plt.plot(x, y, color='C1')
plt.fill_between(x, 0.95*k*x + m, 1.05*k*x + m, alpha=0.2, color='C0')
plt.fill_between(x, 0.9*k*x + m, 1.1*k*x + m, alpha=0.2, color='C0')
plt.fill_between(x, 0.8*k*x + m, 1.2*k*x + m, alpha=0.2, color='C0')

from scipy.stats import gaussian_kde

ker = gaussian_kde([x, y + np.random.normal(size=len(x))])
X, Y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
points = np.vstack((X.ravel(), Y.ravel()))
Z = ker(points).reshape(X.shape)
plt.imshow(np.rot90(Z), extent=[x.min(), x.max(), y.min(), y.max()])

plt.show()
"""

for i, run in enumerate(runs):
    print(f'loading {i+1}/{len(runs)} ', end='')
    with open(path.join(run, 'environ.yaml')) as f:
        env = yaml.safe_load(f)
    cache_filename = path.join(run, '.traj_cache.yaml')
    if not path.exists(cache_filename):
        d = env['parame']['prevision']['max_distance']
        S = [dict(np.load(fn)) for fn in sorted(glob(path.join(run, 'state?????.npz')))]
        #x = np.array([0.0] + [s['completion'] for s in S] + [1.0])
        #y = np.array([0.0] + [s['distance']   for s in S] + [S[-1]['distance']+1e-8])
        x = np.array([s['completion'] for s in S])
        y = np.array([s['distance']   for s in S])
        assert S[-1]['completion'] > 0.99
        assert np.all(np.diff(x[:-1]) >= 0)
        assert np.all(np.diff(y[:-1]) >= 0)
        with open(cache_filename, 'w') as f:
            yaml.dump(dict(d=d, x=x.tolist(), y=y.tolist()), f)
        print('regenerated ', end='')
    else:
        with open(cache_filename) as f:
            dd = yaml.safe_load(f)
        d, x, y = dd['d'], np.array(dd['x']), np.array(dd['y'])
        print('cache ', end='')
    Ds.setdefault(d, [])
    Ds[d].append(Desc(run, d, x, y, f'$d={d:.2f}$'))
    print('d =', d)

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

X = np.linspace(0.05, 0.998, 1000)
for i, d in enumerate(sorted(Ds)):
    F       = [interp1d(ds.x, ds.y, fill_value=(0.0, ds.y.max())) for ds in Ds[d]]
    F_X     = [f(X) for f in F]
    mu_X    = np.mean(F_X, axis=0)
    sigma_X = np.std(F_X, axis=0)
    plt.fill_between(X, mu_X - sigma_X, mu_X + sigma_X, color=f'C{i}', alpha=0.4)
    plt.plot(X, mu_X, color=f'C{i}', alpha=1.0, label=f'$d={d:.2f}, N={len(Ds[d])}$')
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
