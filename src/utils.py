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


def endswith_cycle(seq, *, n, k=2):
    "True if *seq* ends in a k*n long k-cycle"
    subseq = seq[-n*k:]
    if len(subseq) < n*k:
        return False
    pairs = {(i, subseq[j*k + i]) for j in range(n) for i in range(k)}
    return len(pairs) == k


def hsva_to_rgba(hsva):
    """
    Convert hsva values to rgba.

    Parameters
    ----------
    hsva : (..., 4) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgba : (..., 4) ndarray
       Colors converted to RGBA values in range [0, 1]
    """
    hsva = np.asarray(hsva)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsva.shape[-1] != 4:
        raise ValueError("Last dimension of input array must be 4; "
                         "shape {shp} was found.".format(shp=hsva.shape))

    in_shape = hsva.shape
    hsva = np.array(
        hsva, copy=False,
        dtype=np.promote_types(hsva.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    h = hsva[..., 0]
    s = hsva[..., 1]
    v = hsva[..., 2]
    a = hsva[..., 3]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b, a], axis=-1)

    return rgb.reshape(in_shape)
