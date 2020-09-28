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


def is_line_of_sight(mesh, a, b):
    "Is the line a-b free from occlusions"
    return mesh.ray.intersect_any([a], [b - a])[0]


def nearest_visible(r, *, mesh, index, margin=0.1, max_dist=np.inf):
    "Find the nearest neighbors *r_n* of *r* s.t. n-r is not occluded."
    # Find neighbor positions R_nearest
    nearest    = index.nearest(r)[1:]
    R_nearest  = np.array([r_i for r_i, dd_i in nearest])

    # Ray origin is always r, directed towards each neighbor.
    origins    = np.full_like(R_nearest, r)
    directions = R_nearest - r
    dists      = np.linalg.norm(directions, axis=1)

    # Adjust the origin *margin* meters backwards.  so that we test for
    # occlusion a little bit before r, and a little bit after the neighbor's
    # coordinate.
    origins   -= directions/dists[:, None]*margin

    # Only test rays that are less than max_dist long.
    is_near    = dists < max_dist
    origins    =    origins[is_near]
    directions = directions[is_near]
    nearest    = np.array(nearest, dtype=object)[is_near]

    # Run the ray intersection test
    max_dists = dists + 2*margin
    hits      = mesh.ray.intersects_first(origins, directions,
                                          max_dists=max_dists)

    # hits is -1 where a ray did not hit a face.
    is_hit     = hits >= 0
    hit_faces  = hits[is_hit]

    # Compute hit distance for each hitting ray
    hit_dists = np.linalg.norm(mesh.triangles_center[hit_faces] - origins[is_hit], axis=1)

    # Let is_hit be True only when the hit is less than maximum ray distance.
    is_hit[is_hit] &= hit_dists+1e-8 < max_dists[is_hit]

    return [(r_i, dd_i) for r_i, dd_i, is_hit_i in np.c_[nearest, is_hit] if not is_hit_i]


@parame.configurable
def _connect(G, new, r_new, *, index, mesh, force=False,
             max_degree: cfg.param = np.inf,
             max_dist:   cfg.param = np.inf):

    for r_neigh, dd_neigh in nearest_visible(r_new, mesh=mesh, index=index):

        neigh = dd_neigh['node']

        if max_degree <= G.degree(new):
            break

        if max_degree <= G.degree(neigh) and not force:
            continue

        if max_dist < np.linalg.norm(r_new - r_neigh)-1e-8:
            continue

        d = np.linalg.norm(r_new - r_neigh)
        G.add_edge(new, neigh, d=d, rs=[r_new, r_neigh], vs=[new, neigh],
                   skip=False, jump=False)


def insert_random(G, node, *, num_attempts=10):

    mesh  = G.graph['_mesh']
    index = G.graph['_index']
    bbox  = G.graph['_bbox']

    bbox_min, bbox_max = bbox

    for i in range(num_attempts):
        r_node = np.random.uniform(bbox_min, bbox_max, size=(3,))
        index.add(r_node, node=node)
        G.add_node(node, r=r_node, skip=False)
        _connect(G, node, r_node, mesh=mesh, index=index, force=True)
        if G.degree(node) > 0:
            break
        index.remove_nearest(r_node)
        G.remove_node(node)
    else:
        raise RuntimeError('could not place node')

    sensors.update_edge_visibility(G, mesh, save_cache=False)


def extract(G, nodes):
    index = G.graph['_index']
    for node, dd_node in list(G.nodes.data()):
        if node not in nodes:
            log.debug('remove node %d (nbrs: %s)', node, list(G.adj[node]))
            index.remove_nearest(dd_node['r'])
            G.remove_node(node)


def extract_maximal_component(G):
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        log.warn('roadmap is not connected, extracting maximal component')
        log.debug('roadmap component sizes: %s', list(map(len, comps)))
        extract(G, set(max(comps, key=len)))
    else:
        log.info('roadmap is connected, not extracting')


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
        s = cfg['max_dist'] / cfg.get('regular_grid_k', 1)
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
    G.graph['_mesh']  = mesh
    G.graph['_index'] = index
    G.graph['_bbox']  = bbox
    #G.graph['_octree'] = octree

    for i, coord in enumerate(coords):
        index.add(coord, node=i)
        G.add_node(i, r=coord, skip=False)

    log.debug('average number of nodes in each spatial index bin: %.2f',
              np.mean([len(bin) for bin in index.iter_bins()]))

    for new, r_new in enumerate(coords):

        _connect(G, new, r_new, mesh=mesh, index=index)

        if ((new+1) % (len(coords)//10)) == 0:
            log.debug(f'connected {new+1:5d} nodes, mean degree: %.2f',
                      np.mean([G.degree(v) for v in range(new+1)]))
            gui.update_roadmap(G)
            gui.wait_draw()

    extract_maximal_component(G)

    sensors.update_edge_visibility(G, mesh)

    return G


@parame.configurable
def update_jumps(G, *, seen, active=(),
                 skip_max_faces:  cfg.param = 0,
                 max_degree_jump: cfg.param = 36):

    vis_faces = G.graph['vis_faces']
    vis_unseen = vis_faces & ~seen

    # Mark edges skip if they see no unseen faces
    for u, v, dd_uv in list(G.edges.data()):
        if dd_uv['jump']:
            G.remove_edge(u, v)
        num_unseen_uv = np.count_nonzero(dd_uv['vis_faces'] & vis_unseen)
        dd_uv['skip'] = num_unseen_uv <= skip_max_faces

    for n, dd_n in G.nodes.data():
        nbrs_skip = [dd_m['skip'] for dd_m in G._adj[n].values()]
        dd_n['skip'] = all(nbrs_skip)
        dd_n['jump'] = any(nbrs_skip) and not dd_n['skip']

    # Nodes marked skip should not have any jump edges: all their edges are skip edges.
    # Nodes marked jump should have a jump edge to every other jump node.
    Jump = NodeDataMap(G, 'jump')

    nodes = {u for u in G if Jump[u] or u in active}
    edges = [uv for uv in combinations(nodes, 2) if uv not in G.edges]
    np.random.shuffle(edges)

    log.info('%d jump nodes, adding %d edges', len(nodes), len(edges))

    # Need to clear previous shortest paths memo
    G.graph['_shortest_path_memo'] = {}

    for u, v in edges:
        if {u, v} & active or max(G.degree(u), G.degree(v)) < max_degree_jump:
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


def _shortest_path_memoized(G, u, v):
    "Shortest path from u to v in G, memoizes all shortest paths from u."
    _shortest_path_memo = G.graph.setdefault('_shortest_path_memo', {})

    # Check first if we already know the reverse (i.e. v to u)
    if v in _shortest_path_memo:
        if u not in _shortest_path_memo[v]:
            raise nx.NetworkXNoPath((u, v))
        return _shortest_path_memo[v][u][::-1]

    if u not in _shortest_path_memo:
        _shortest_path_memo[u] = nx.shortest_path(G, u, weight=_weightfunc)

    if v not in _shortest_path_memo[u]:
        raise nx.NetworkXNoPath((u, v))

    return _shortest_path_memo[u][v]


def _add_jump(G, u, v, *, path=None, **kw):
    "Add jump u-v in G"

    if (u, v) in G.edges:
        log.warn('adding a jump edge between neighbors, bad idea.')

    R         = NodeDataMap(G, 'r')
    Vs        = EdgeDataMap(G, 'vs')
    Vis_faces = EdgeDataMap(G, 'vis_faces')

    if path is None:
        path = _shortest_path_memoized(G, u, v)

    # Turn the path into its non-jump edge sequence.
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
