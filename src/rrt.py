import logging
from collections import deque

import numpy as np
import networkx as nx

import parame
import spatial
import gui
from tree import Tree


cfg = parame.Module(__name__)
log = logging.getLogger(__name__)

norm = np.linalg.norm

unknown_is_occupied = cfg.get('unknown_is_occupied', True)


def is_traversable(octree, start, goal):
    # NOTE Should do test with a radius
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


def is_occupied(octree, point):
    node = octree.search(point)
    if not node:
        return unknown_is_occupied
    return octree.isNodeOccupied(node)


def new(position, *, root=0, d=0.0):
    T = Tree()
    T.add_node(root, r=position, d=d)
    T.graph['root'] = root
    return T


#def subtree(T, new_root):
#    S = type(T)()
#    node_map = {}
#    m = lambda u: node_map.setdefault(u, len(node_map))
#    S.graph = T.graph.copy()
#    S.root = m(new_root)
#    stack = [new_root]
#    while stack:
#        u = stack.pop()
#        mu = m(u)
#        S._node[mu] = T._node[u]
#        S._succ[mu] = {m(v): dd.copy() for v, dd in T._succ[u].items()}
#        for mv, dd in S._succ[mu].items():
#            S._pred[mv] = {mu: dd}
#        stack.extend(T._succ[u])
#    return S


def best_node(roadmap):
    # Find the branch with best cost
    d_root        = roadmap.nodes[roadmap.root]['d']
    vis_area_root = roadmap.nodes[roadmap.root]['vis_area']

    def cost(u, data_u):
        vis_area_u = data_u['vis_area']
        min_dist_u = data_u['min_dist']
        d_u        = data_u['d']
        return (vis_area_u - vis_area_root)
        return np.sqrt(vis_area_u - vis_area_root) - 0.4*(d_u - d_root) #if data_u['min_dist'] > 5e-2 else -np.inf

    node, data_node = max(roadmap.nodes.data(), key=lambda pair: cost(*pair))
    return node

    datapoints = [(node, np.sqrt(data['vis_area'] - vis_area_root), (data['d'] - d_root))
                  for node, data in roadmap.nodes.data()]
    dps = np.array(datapoints)
    x = dps[:, 2]
    A = np.c_[x, np.ones(len(x))]
    y = dps[:, 1]
    (k, m), res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    #k += 0.05
    #from matplotlib import pyplot as plt
    #plt.hist2d(dps[:, 0], dps[:, 1], bins=100)
    #plt.scatter(x, y)
    #plt.plot(x, k*x + m)
    #plt.scatter(x, y - (k*x + m))
    #plt.show()
    #datapoints = [(n, sqrvg, dd)
    #              for n, sqrvg, dd in datapoints if sqrvg > 0]
    print(datapoints[np.argmax(y - (k*x + m))])
    node = datapoints[np.argmax(y - (k*x + m))][0]

    return node


def extract_path(T, path):
    "Extract only *path* in *T* into a new graph and renumber nodes."

    # Find new labels for the copied nodes
    path_S  = np.arange(len(path))
    path_TS = np.c_[path, path_S]

    # Copy graph metadata
    S       = type(T)()
    S.graph = T.graph.copy()
    S.root  = path_S[0]

    # Copy nodes
    for Tu, Su in path_TS:
        S._pred[Su] = {}
        S._succ[Su] = {}
        S._node[Su] = T._node[Tu].copy()

    # Copy edges
    for (Tu, Su), (Tv, Sv) in zip(path_TS[:-1], path_TS[1:]):
        S._succ[Su][Sv] = T._succ[Tu][Tv].copy()
        S._pred[Sv][Su] = T._pred[Tv][Tu].copy()

    return S


"""
def reroot(T, new_root):
    S = type(T)()
    S.graph = T.graph.copy()
    S.root = new_root
    S._node = {u: dd_u for u, dd_u in T._node.items()}
    S._succ = {u: {v: dd_uv for v, dd_uv in succ_u.items()} for u, succ_u in T._succ.items()}
    S._pred = {v: {u: dd_uv for u, dd_uv in pred_v.items()} for v, pred_v in T._pred.items()}
    S._adj = S._succ

    path = S.path(S.root)
    for u, v in zip(path[:-1], path[1:]):
        dd_uv = S._succ[u][v]
        del S._succ[u][v], S._pred[v][u]
        S._succ[v][u] = S._pred[u][v] = dd_uv

    # Adjust d and vis_faces
    for u, v in [(path[-1], path[-2]), *nx.bfs_edges(S, source=path[-2])]:
        S._node[v]['d'] = S._node[u]['d'] + S._node[u]['d'] - S._node[v]['d']

    return S
"""



"""
def pg(T):
    for u in T:
        succ = list(T.successors(u))
        print(u, T.nodes[u]['d'], '->', ', '.join(succ))

t = Tree()
t.add_nodes_from('abcde')
t.add_edge('a', 'b')
t.add_edge('a', 'd')
t.add_edge('b', 'c')
t.add_edge('d', 'e')
t.nodes['a']['d'] = 0.0
for u,v in nx.edge_bfs(t, source='a'):
    t.nodes[v]['d'] = t.nodes[u]['d'] + 1.0
pg(t)
print()
t_ = reroot(t, 'b')
pg(t_)

raise SystemExit
"""


@parame.configurable
def extend(tree, *, octree, bbox=None,
           bin_size:          cfg.param = 0.50,
           max_dist:          cfg.param = None,
           num_nodes_max:     cfg.param = 1200,
           #num_child_max:     cfg.param = 10,
           #num_neighbors_max: cfg.param = 12,
           num_child_max:     cfg.param = np.inf,
           num_neighbors_max: cfg.param = 200,
           num_samples:       cfg.param = 5000,
           num_attempts_max:  cfg.param = 15000):

    if bbox is None:
        bbox = np.array([octree.getMetricMin(),
                         octree.getMetricMax()])

    log.info(f'building rrt (max {num_nodes_max} nodes)')
    msglines = str(bbox).splitlines()
    log.debug(f'bounding box min: {msglines[0]}')
    log.debug(f'bounding box max: {msglines[1]}')
    bbox_min, bbox_max = bbox

    index = spatial.Index(bbox_min, bbox_max, bin_size=bin_size)
    log.debug(f'spatial index bin size:  {index.bin_size}')
    log.debug(f'spatial index bin count: {index.num_bins}')

    for node, node_data in tree.nodes.data():
        index.add(node_data['r'], node=node, d=node_data['d'])

    log.debug('index constructed')

    def distfun(x, y, data=None):
        return data['d'] + norm(x - y)

    num_inserts  = 0
    num_rewires  = 0
    num_attempts = 0

    # Prepare a set of possible sample points. Not sure this is any faster.
    coords = np.random.uniform(bbox_min, bbox_max, size=(num_samples, 3))
    coords = [coord for coord in coords if not is_occupied(octree, coord)]
    cands = deque(coords)

    log.debug('generated candidates')

    while cands and len(tree) < num_nodes_max and num_attempts < num_attempts_max:

        num_attempts += 1

        # Sampled point r.
        r_new = cands.popleft()

        # Node label.
        new = len(tree)

        # Find the neighbor set in terms of total distance.
        neighbors = index.nearest(r_new, distfun=distfun)

        # If no neighbors, look at candidate again later when there may be.
        if not neighbors:
            cands.append(r_new)
            continue

        # Some regions get oversampled, don't consider these
        if len(neighbors) > num_neighbors_max:
            continue

        # Find the closest non-colliding node.
        try:
            r_parent, parent_data = next(
                (r_neigh, neigh_data)
                for (r_neigh, neigh_data) in neighbors
                if is_traversable(octree, r_new, r_neigh))
        except StopIteration:
            # No neighbor is eligible, try again later when one may be.
            cands.append(r_new)
            continue

        new_parent = parent_data['node']
        d_parent   = parent_data['d']

        # Restrict number of children to avoid oversampling. Do not reinsert
        # into candidate queue.
        if tree.degree[new_parent] > num_child_max:
            continue

        # Move new within maximum distance of its parent.
        if max_dist is not None:
            direction = r_new - r_parent
            r_new = r_parent + min(max_dist/norm(direction), 1.0)*direction
            neighbors = index.nearest(r_new, distfun=distfun)

        # Compute d for new node.
        d_new = d_parent + norm(r_new - r_parent)

        # Insert me like one of your French girls.
        tree.add_node(new, r=r_new, d=d_new)
        tree.add_edge(new_parent, new)
        index.add(r_new, node=new, d=d_new)

        num_inserts += 1

        # We may have just inserted a better parent for some existing nodes,
        # find out and if so, rewire them.
        for r_neigh, neigh_data in neighbors:

            neigh        = neigh_data['node']
            neigh_parent = tree.parent(neigh)
            d_neigh      = neigh_data['d']

            # Suggested d_neigh, if rewired.
            d_neigh_ = d_new + norm(r_new - r_neigh)

            if d_neigh_ < d_neigh and is_traversable(octree, r_new, r_neigh):

                tree.remove_edge(neigh_parent, neigh)

                neigh_parent = new
                d_neigh      = d_neigh_

                tree.add_edge(neigh_parent, neigh)
                tree.nodes[neigh]['d']   = d_neigh
                neigh_data['d']          = d_neigh

                num_rewires += 1

        # Log some running stats.
        if (len(tree) % (num_nodes_max//10)) == 0:
            log.debug(f'inserted {num_inserts:5d} nodes, avg out degree: %.2f edges/node',
                      np.mean([len(tree.succ[v]) for v in tree]))
            log.debug(f'num nodes per spatial bin: %.2f nodes/bin',
                      np.mean([len(bin) for bin in index.iter_bins()]))
            gui.update_roadmap(tree)
            gui.wait_draw()

    log.debug('finished, graph size %d (%d inserted, %d attempts, %d rewires)',
              len(tree), num_inserts, num_attempts, num_rewires)
