r"""Planning Graph for Exploration

We start out with a graph G = (V, E) where V is a set of states. These states
are either sampled randomly, or simply a regular grid. For each state v in V
then, connect it to each u in V if u-v is traversible in a without collision.

In our case, the state space is simply positions in R^3, and the traversibility
is equivalent to non-occlusion.

Having formed this graph, we then compute the exact set of faces visible along
each edge in the graph. See `sensors.py` for details.

What makes our graph planner special is that it adds new edges as the
exploration proceeds. Consider a graph G with edges in series,

    v_0 - v_1 - v_2 - v_3 - ... - v_k - ... - v_n

then assume the exploration started at v_0 and continued in a straight fashion
to v_k. Clearly, there is nothing to see "behind" it, i.e. the states v_i : i
<= k have all been visited. It is therefore acceptable to simply not search
backwards in this case. We can hence remove the edges we pass as we explore.

Assume instead that the exploration started at v_1, now the same is true, but
we must keep v_0 - v_1. However, this would create a disconnected graph where
v_0 - v_1 is its own component. So, we insert a new edge v_1 - v_k. In this way
v_1 - v_2 - ... - v_k is replaced by v_1 - v_k:

    v_0 - v_1 - v_k - ... - v_n

Consider a less trivial graph:

         v_00 ---- v_01 ---- v_02 ---- ...
        /    \    /    \    /    \    /    \
    v_10 ---- v_11 ---- v_12 ---- v_13 ---- ...
        \    /    \    /    \    /    \    /
         v_20 ---- v_21 ---- v_22 ---- ...

Consider the path v_00-v_10-v-20. We can't remove v_10's edges as there is an
unvisited edge v_10-v_11. A graph search must be able to find this edge.
However, we can insert a "jump" edge v_00-v20, to help the graph search jump to
more interesting states.  Once v_10-v_11 is visited, we can remove all edges
connected to v_10.

The terminology is skip and jump: a skip edge is marked skip because it
sees no unseen faces, and a jump edge is a combination of several skip edges.
Jump edges are therefore always skip edges.

"""

import logging

import numpy as np
import networkx as nx
from networkx.utils import pairwise

import gui
import parame
import spatial
import sensors
from tree import NodeDataMap, EdgeDataMap


cfg = parame.Module(__name__)
log = logging.getLogger(__name__)

unknown_is_occupied = cfg.get('unknown_is_occupied', True)


def is_traversable(octree, start, goal):
    keys = octree.computeRayKeys(start, goal)
    if keys is None:
        return False
    for key in keys:
        node = octree.search(key)
        if not node:
            if unknown_is_occupied:
                return False
            continue
        elif octree.isNodeOccupied(node):
            return False
    return True


def is_occupied(point, *, octree):
    node = octree.search(point)
    if not node:
        return unknown_is_occupied
    return octree.isNodeOccupied(node)


@parame.configurable
def new(*, mesh, octree, bbox=None, nodes={},
        bin_size:         cfg.param = 1.5,
        regular_grid:     cfg.param = False,
        num_nodes_max:    cfg.param = 2000,
        z_bounds:         cfg.param = None):

    log.info(f'building prm (max {num_nodes_max} nodes)')

    if bbox is None:
        bbox = np.array([octree.getMetricMin(),
                         octree.getMetricMax()])

    if z_bounds is not None:
        bbox = bbox.copy()
        assert len(z_bounds) == 2, f'{z_bounds!r} is a pair of numbers'
        bbox[:, 2] = z_bounds

    msglines = str(bbox).splitlines()
    log.debug(f'bounding box min: {msglines[0]}')
    log.debug(f'bounding box max: {msglines[1]}')
    bbox_min, bbox_max = bbox

    if not regular_grid:
        log.debug('sampling random coordinates')
        coords = np.random.uniform(bbox_min,
                                   bbox_max,
                                   size=(2*num_nodes_max, 3))
    else:
        s = regular_grid
        log.debug(f'creating regular grid coordinates (side: {s})')
        # In a triangularly tessellated grid of side s, the spacing needs to be
        # adjusted by sqrt(3)/2.
        coords = np.mgrid[bbox_min[0]:bbox_max[0]:np.sqrt(3)/2*s,
                          bbox_min[1]:bbox_max[1]:s,
                          bbox_min[2]:bbox_max[2]:np.sqrt(3)/2*s]
        coords[1, ::2, :,   :] += s/2
        coords[1,   :, :, ::2] += s/2
        coords = coords.reshape(3, -1).T.copy()

    log.debug(f'{len(coords)} candidate coordinates')

    coords = coords[~np.apply_along_axis(is_occupied, 1, coords, octree=octree)]
    log.debug(f'{len(coords)} coordinates remain after removing occupied')

    if len(coords) > num_nodes_max:
        idxs   = np.random.choice(len(coords), size=num_nodes_max, replace=False)
        coords = coords[idxs, :]
        log.debug(f'{len(coords)} coordinates subsampled')

    index = spatial.Index(bbox_min, bbox_max, bin_size=bin_size)
    log.debug(f'spatial index bin size:  {index.bin_size}')
    log.debug(f'spatial index bin count: {index.num_bins}')

    G = nx.Graph()
    G.graph['index']  = index
    G.graph['octree'] = octree

    for i, coord in enumerate(coords):
        index.add(coord, node=i)
        G.add_node(i, r=coord, skip=False)

    log.debug('average number of nodes in each spatial index bin: %.2f',
              np.mean([len(bin) for bin in index.iter_bins()]))

    for new, r_new in enumerate(coords):

        _connect(G, new, r_new, index, octree)

        if ((new+1) % (len(coords)//10)) == 0:
            log.debug(f'connected {new+1:5d} nodes, mean degree: %.2f',
                      np.mean([G.degree(v) for v in range(new+1)]))
            gui.update_roadmap(G)
            gui.wait_draw()

    for node, r_node in nodes.items():
        index.add(r_node, node=node)
        G.add_node(node, r=r_node, skip=False)
        _connect(G, node, r_node, index, octree, force=True)

    sensors.update_edge_visibility(G, mesh)

    return G


def insert(G, r_new):
    index  = G.graph['index']
    octree = G.graph['octree']
    new    = len(G)

    index.add(r_new, node=new)
    G.add_node(new, r=r_new, skip=False)
    _connect(G, new, r_new, index, octree, force=True)

    return new


@parame.configurable
def _connect(G, new, r_new, index, octree, *, force=False,
             num_adjacent_max: cfg.param = np.inf,
             max_dist: cfg.param = np.inf):

    nbrs = [(data['node'], r_neigh)
            for r_neigh, data in index.nearest(r_new)[1:]
            if is_traversable(octree, r_new, r_neigh)]

    for neigh, r_neigh in nbrs:

        if num_adjacent_max < G.degree(new):
            break

        if num_adjacent_max < G.degree(neigh) and not force:
            continue

        if max_dist < np.linalg.norm(r_new - r_neigh)-1e-8:
            continue

        d = np.linalg.norm(r_new - r_neigh)
        G.add_edge(new, neigh, d=d, rs=[r_new, r_neigh], vs=[new, neigh], skip=False)


def update_skips(G, *, seen, keep=()):

    # Mark edges skip if they see no unseen faces
    for u, v, dd_uv in list(G.edges.data()):
        dd_uv['skip'] = np.count_nonzero(dd_uv['vis_faces'] & ~seen) < 10

    for n, dd_n in G.nodes.data():
        dd_n['skip'] = all(dd_m['skip'] for dd_m in G._adj[n].values())

    Skip = NodeDataMap(G, 'skip')

    # Find triplets u-v-w where u-v and v-w have been visited, and replace them
    # by the skip edge u-w (unless it already exists). This does not check
    # edges of nodes marked skip. Therefore we mark the nodes skip after.
    n_combined = 1
    while n_combined > 0:

        n_combined = 0
        n_exmanied = 0

        for u, v, dd_uv in list(G.edges.data()):

            skip_uv = dd_uv['skip']

            if Skip[u] or Skip[v] or not skip_uv:
                continue

            for v, w, dd_vw in list(G.edges(v, data=True)):

                if u == w or Skip[w] or (u, w) in G.edges:
                    continue

                skip_vw = dd_vw['skip']

                if skip_uv and skip_vw:
                    combine_triplet(G, u, v, w, skip=True)
                    #if len(dd_uv['vs']) > 2: G.remove_edge(u, v)
                    #if len(dd_vw['vs']) > 2: G.remove_edge(v, w)
                    #assert not np.any(G.edges[u, w]['vis_faces'] & ~seen)
                    n_combined += 1

        if n_combined:
            log.info('combined %d edges', n_combined)

    # Mark nodes skip (and remove skip edges) when their edges are skip
    n_removed = 0
    for n, dd_n in G._node.items():

        if n in keep: # or dd_n['skip']:
            continue

        nbrs = G._adj[n]
        skip_edges = [(n, m) for m, dd_m in nbrs.items() if len(dd_m['vs']) > 2]

        if skip_edges and all(dd_m.get('skip') for dd_m in nbrs.values()):
            log.debug('removing %d skip edges from %s', len(skip_edges), n)
            n_removed += len(skip_edges)
            G.remove_edges_from(skip_edges)
            dd_n['skip'] = True

    if n_combined:
        log.info('removed %d edges', n_combined)

    log.info(f'{len(G.edges())} edges after updating skips')


def combine_triplet(G, u, v, w, **kw):
    "Take a triplet of nodes u-v-w in G and create u-v, bypassing v."

    R         = NodeDataMap(G, 'r')
    Vs        = EdgeDataMap(G, 'vs')
    Vis_faces = EdgeDataMap(G, 'vis_faces')

    # Find the shortest path in G. This will use skip edges.
    path   = nx.shortest_path(G, u, w, weight=lambda u, v, dd_uv: dd_uv['d'] if dd_uv['skip'] else None)

    # Turn the path into an actually traversible path.
    vs_uw  = []
    for p, q in pairwise(path):
        vs_pq = Vs[p, q]
        vs_pq = vs_pq if vs_pq[0] == p else vs_pq[::-1]
        assert vs_pq[0] == p and vs_pq[-1] == q
        vs_uw.extend(vs_pq[:-1])
    vs_uw.append(w)

    assert vs_uw[0] == u and vs_pq[-1] == w

    rs_uw  = [R[s] for s in vs_uw]
    vis_uw = np.any([Vis_faces[s, s_] for s, s_ in pairwise(vs_uw)], axis=0)
    d_uw   = np.sum(np.linalg.norm(np.diff(rs_uw, axis=0), axis=1))
    G.add_edge(u, w, d=d_uw, vis_faces=vis_uw, rs=rs_uw, vs=vs_uw, **kw)


