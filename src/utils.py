import threading
import logging

import numpy as np


log = logging.getLogger(__name__)


class Thread(threading.Thread):
    def run(self):
        log.debug('[%s] thread started', self.name)
        try:
            self._return = self._target(*self._args, **self._kwargs)
        except Exception:
            log.exception('[%s] thread aborted by exception', self.name)
        else:
            log.debug('[%s] thread finished', self.name)
        finally:
            del self._target, self._args, self._kwargs


def threadable(f):
    fn = f'{f.__module__}.{f.__name__}'
    def thread(*args, daemon=False, **kwds):
        name = threading._newname(template=f'{fn}:%d')
        return Thread(target=f, args=args, kwargs=kwds,
                      name=name, daemon=daemon)
    f.thread = thread
    return f


def graph_md5(G):
    import hashlib
    hasher = hashlib.md5()
    keysets = {'graph': {'graph': G.graph},
               'node': {u: dd for u, dd in G.nodes.data()},
               'adj': {(u, v): dd for u, v, dd in G.edges.data()}}
    for prefix, kset in keysets.items():
        for k, d in kset.items():
            for kd, vd in d.items():
                if not kd.startswith('_'):
                    data = f'{prefix}[{k!r}][{kd!r}] = {vd!r}\n'
                    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()


def format_duration(secs):
    units = [('s', 60), ('m', 60), ('h', 24), ('d', 7), ('w', np.inf)]
    v = secs
    L = []
    while units:
        unit, size = units.pop(0)
        digit = v % size
        if digit > 1e-18:
            L.append(f'{digit:.0f}{unit}')
        v -= digit
        v /= size
        if v < 1:
            break
    return ''.join(L[::-1])
