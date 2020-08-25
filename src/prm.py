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
from itertools import combinations

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


def sample_coord(bbox_min, bbox_max, octree):
    for i in range(100):
        coord = np.random.uniform(bbox_min, bbox_max, size=(3,))
        if not is_occupied(coord, octree=octree):
            return coord
    raise RuntimeError('rejection sampling ran out of attempts')


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
    #G.graph['_index']  = index
    #G.graph['_octree'] = octree

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

    sensors.update_edge_visibility(G, mesh)

    for node, r_node in nodes.items():
        if r_node == 'random':
            r_node = sample_coord(bbox_min, bbox_max, octree)
        index.add(r_node, node=node)
        G.add_node(node, r=r_node, skip=False)
        _connect(G, node, r_node, index, octree, force=True)

    sensors.update_edge_visibility(G, mesh, save_cache=False)

    return G


def insert(G, r_new):
    index  = G.graph['_index']
    octree = G.graph['_octree']
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
        G.add_edge(new, neigh, d=d, rs=[r_new, r_neigh], vs=[new, neigh],
                   skip=False, jump=False)


@parame.configurable
def update_jumps(G, *, seen, active=(),
                 skip_max_faces: cfg.param = 4):

    vis_faces = G.graph['vis_faces']
    vis_unseen = vis_faces & ~seen

    # Mark edges skip if they see no unseen faces
    for u, v, dd_uv in list(G.edges.data()):
        if dd_uv['jump']:
            continue
        unseen_faces_uv = np.count_nonzero(dd_uv['vis_faces'] & vis_unseen)
        dd_uv['skip'] = unseen_faces_uv < skip_max_faces

    for n, dd_n in G.nodes.data():
        nbrs_skip = [dd_m['skip'] for dd_m in G._adj[n].values()]
        dd_n['skip'] = all(nbrs_skip)
        dd_n['jump'] = any(nbrs_skip) and not dd_n['skip']

    # Nodes marked skip should not have any jump edges: all their edges are skip edges.
    # Nodes marked jump should have a jump edge to every other jump node.
    Jump = NodeDataMap(G, 'jump')

    nodes = {u for u in G if Jump[u] or u in active}
    edges = {uv for uv in combinations(nodes, 2) if uv not in G.edges}

    log.info('%d jump nodes, adding %d edges', len(nodes), len(edges))

    _shortest_path_memo.clear()
    for u, v in edges:
        try:
            _add_jump(G, u, v, skip=True, jump=True)
        except nx.NetworkXNoPath:
            log.warn(f'jump edge insertion failed, no path {u}-{v} found')

    # Mark nodes skip (and remove skip edges) when their edges are skip
    n_removed = 0
    for n, dd_n in G._node.items():

        if n in active:
            continue

        nbrs = G._adj[n]
        jump_edges = {(n, m) for m, dd_m in nbrs.items() if dd_m['jump']}

        if jump_edges and dd_n['skip']:
            log.debug('removing %d jump edges from %s', len(jump_edges), n)
            n_removed += len(jump_edges)
            G.remove_edges_from(jump_edges)

    if n_removed:
        log.info('removed %d edges', n_removed)

    log.info(f'{len(G.edges())} edges after updating jumps')


def _weightfunc(u, v, dd):
    "Edge weight = distance for visited non-jump edges, otherwise None."
    if dd['skip'] and not dd['jump']:
        return dd['d']


_shortest_path_memo = {}

def _add_jump(G, u, v, **kw):
    "Add jump u-v in G"

    R         = NodeDataMap(G, 'r')
    Vs        = EdgeDataMap(G, 'vs')
    Vis_faces = EdgeDataMap(G, 'vis_faces')

    # Find shortest path from u to v in G, and cache the result.
    if u not in _shortest_path_memo:
        _shortest_path_memo[u] = nx.shortest_path(G, u, weight=_weightfunc)

    if v not in _shortest_path_memo[u]:
        raise nx.NetworkXNoPath((u, v))

    path = _shortest_path_memo[u][v]

    #path = nx.shortest_path(G, u, v, weight=_weightfunc)

    # Turn the path into an actually traversible path.
    vs_uv  = []
    for p, q in pairwise(path):
        vs_pq = Vs[p, q]
        vs_pq = vs_pq if vs_pq[0] == p else vs_pq[::-1]
        assert vs_pq[0] == p and vs_pq[-1] == q
        vs_uv.extend(vs_pq[:-1])
    vs_uv.append(v)

    assert vs_uv[0] == u and q == v and vs_pq[-1] == v

    rs_uv  = [R[s] for s in vs_uv]
    vis_uv = np.any([Vis_faces[s, s_] for s, s_ in pairwise(vs_uv)], axis=0)
    d_uv   = np.sum(np.linalg.norm(np.diff(rs_uv, axis=0), axis=1))
    G.add_edge(u, v, d=d_uv, vis_faces=vis_uv, rs=rs_uv, vs=vs_uv, **kw)
