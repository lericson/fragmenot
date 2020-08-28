"The Bin Lading Spatial Index"


import numpy as np


def adjacent_bin_deltas(bin_range):
    adj_steps = np.arange(-bin_range, bin_range+1)
    deltas = np.array([[dz, dy, dx, (dz**2+dy**2+dx**2)]
                       for dz in adj_steps
                       for dy in adj_steps
                       for dx in adj_steps])
    deltas = deltas[deltas[:, 3] <= bin_range**2]
    deltas = deltas[np.argsort(deltas[:, 3])]
    return deltas[:, 0:3].copy()


def bindex(coord, origin, size, num_bins):
    "Bin Index - BinIndex - BinDex"
    coord_norm  = coord - origin
    coord_norm /= size
    return np.floor(coord_norm*num_bins).astype(int)


def euclidean_dist(x, y, data):
    return np.linalg.norm(x - y)


class Index():
    """A basic spatial index

    Each entry is a 3D coordinate, along with some associated data. The
    associated data is not indexed, though I guess it could be.

    The index is just 3D binning of the coordinates. Specify num_bins for each
    dimension, and you will have that many bins to put your stuff into - the
    bin size will be determined by the bounding box. Specify bin_size for each
    dimension, and the number of bins will be computed from the bounding box
    instead.

    Queries for nearest entries are satisfied by finding the closest bins in a
    spherical fashion. bin_search_range controls how many bins away from the
    requested coordinate to search. Note that you never need to increase this
    number unless you want a more faithful spherical search. Rather, you can
    control the size of the bins.
    """

    def __init__(self, bounds_min, bounds_max, *, num_bins=None, bin_size=None,
                 bin_search_range=3):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.origin     = bounds_min
        self.size       = bounds_max - self.bounds_min

        if num_bins is not None and bin_size is not None:
            raise ValueError('pass either num_bins or bin_size, not both')
        elif num_bins is not None:
            bin_size = self.size/num_bins
        elif bin_size is not None:
            num_bins = np.ceil(self.size/bin_size).astype(int)

        num_bins0, num_bins1, num_bins2 = num_bins
        self.num_bins   = num_bins
        self.bin_size   = bin_size

        self.bindeltas  = adjacent_bin_deltas(bin_search_range)
        self.bins       = [[[[]
                             for x in range(num_bins2)]
                            for y in range(num_bins1)]
                           for z in range(num_bins0)]

    def bindex(self, coord):
        return bindex(coord, self.origin, self.size, self.num_bins)

    def nearest(self, coord, *, distfun=euclidean_dist, n=None):
        bidx = self.bindex(coord)
        cands = []
        for bdelta in self.bindeltas:
            bidx0_, bidx1_, bidx2_ = bidx_ = bidx + bdelta
            if np.all(0 <= bidx_) and np.all(bidx_ < self.num_bins):
                cands += self.bins[bidx0_][bidx1_][bidx2_]
        cands.sort(key=lambda pair: distfun(coord, *pair))
        return cands

    def add(self, coord, **kwargs):
        bidx0, bidx1, bidx2 = self.bindex(coord)
        L = self.bins[bidx0][bidx1][bidx2]
        L.append((coord, kwargs))

    def remove_nearest(self, coord, *, distfun=euclidean_dist):
        bidx0, bidx1, bidx2 = self.bindex(coord)
        L = self.bins[bidx0][bidx1][bidx2]
        L.pop(min(range(len(L)), key=lambda i: distfun(coord, *L[i])))

    def iter_bins(self):
        return (b for layer in self.bins for row in layer for b in row)
