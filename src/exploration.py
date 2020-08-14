import logging
from os import makedirs, getpid, path
from dataclasses import dataclass

import trimesh
import numpy as np
import networkx as nx
from networkx.utils import pairwise

import gui
import prm
import parame
import tree_search
from utils import threadable


log = logging.getLogger(__name__)
cfg = parame.Module(__name__)


@dataclass(frozen=True)
class State():
    mesh: trimesh.Trimesh
    position: np.ndarray
    seen_faces: np.ndarray
    roadmap: nx.Graph
    node: int             = None
    search_tree: nx.Graph = None
    trajectory: list      = None

    @classmethod
    @parame.configurable
    def new(cls, *, mesh, octree,
            start_position: cfg.param = (0, 0, 2)):
        pos     = np.asarray(start_position, dtype=float)
        seen    = np.zeros(mesh.faces.shape[0], dtype=bool)
        roadmap = prm.new(mesh=mesh, octree=octree, nodes={'start': pos})
        return cls(mesh, pos, seen, roadmap, 'start', trajectory=[])

    @property
    def save_dict(self):
        return dict(pos=self.position, seen=self.seen_faces,
                    roadmap=self.roadmap, trajectory=self.trajectory)

    @property
    def distance(self):
        if len(self.trajectory) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1))

    @property
    def completion(self):
        visible    = self.roadmap.graph['vis_faces']
        seen       = self.seen_faces
        area_faces = self.mesh.area_faces
        return np.sum(area_faces[seen]) / np.sum(area_faces[visible])

    def update_gui(self):
        visible = self.roadmap.graph['vis_faces']
        gui.update_roadmap(self.roadmap)
        gui.update_position(self.position)
        gui.update_trajectory(self.trajectory)
        gui.update_vis_faces(visible=visible, seen=self.seen_faces)


@threadable
def step(state):

    mesh    = state.mesh
    pos     = state.position
    seen    = state.seen_faces
    roadmap = state.roadmap
    node    = state.node
    traj    = state.trajectory
    stree   = state.search_tree

    #if len(roadmap) < 2:
    #    rrt.extend(roadmap, octree=octree, bbox=bbox)

    #target     = rrt.best_node(roadmap)
    #data_target = roadmap.nodes[target]
    #vis_target = data_target['vis_faces']
    #if target == roadmap.root:
    #    log.info('planner suggested root node; exploration complete?')
    #    target = np.random.choice(len(roadmap))
    #path       = roadmap.path(roadmap.root, target)

    roadmap_ = roadmap.copy()
    del roadmap

    prm.update_skips(roadmap_, seen=seen, keep={node})
    gui.update_roadmap(roadmap_)
    gui.wait_draw()

    #if stree is None:
    stree = tree_search.new(start=node, seen=seen)
    stree = tree_search.expand(stree, roadmap=roadmap_, mesh=mesh)
    path, vis_path, stree_ = tree_search.best_path(stree)

    if len(path) > 1:
        log.info('chose path: %s', path)

        # node_ is the next node in the chosen path
        node_ = path[1]

        # Construct next state (hence the _ suffixes)
        pos_       = roadmap_.nodes[node_]['r'].copy()
        seen_      = seen | roadmap_.edges[node, node_]['vis_faces']
        #seen_      = data_v['vis_faces'].copy()
        #roadmap_   = rrt.new(pos_) # NOTE should set root d = data_v['d']
        #roadmap_   = rrt.reroot(roadmap, node_)
        #roadmap_   = rrt.extract_path(roadmap, path[1:])
        #roadmap_   = rrt.new(pos_, d=data_v['d']) # NOTE should set root d = data_v['d']

        rs = roadmap_.edges[node, node_]['rs']
        rs = rs if np.allclose(rs[0], roadmap_.nodes[node]['r']) else rs[::-1]
        traj_ = traj + rs

    else:
        log.warn('no path found in planning step, standing still')
        node_      = node
        pos_       = pos
        seen_      = seen
        traj_      = traj

    del node, pos, seen, traj, state

    state_ = State(mesh=mesh,
                   position=pos_,
                   seen_faces=seen_,
                   roadmap=roadmap_,
                   search_tree=stree_,
                   node=node_,
                   trajectory=traj_)

    state_.update_gui()
    gui.hilight_roadmap_edges(roadmap_, pairwise(path))
    gui.hilight_vis_faces(vis_path & ~seen_)

    log.info('new state %s: %s', node_, pos_)
    log.info('explored %.2f%% of visible faces', 100*state_.completion)

    return state_


@parame.configurable
def output_path(*args, create=True,
                output_dir: cfg.param = f'runs{path.sep}run_{getpid()}'):
    if create:
        makedirs(output_dir, exist_ok=True)
    return path.join(output_dir, *args)


@threadable
@parame.configurable
def run(*, octree, mesh, state=None,
        num_steps: cfg.param = 2000):

    for i in range(num_steps):

        if state is None:
            state = State.new(octree=octree, mesh=mesh)
            state.update_gui()

        else:
            state = step(state)

        gui.show_message(f'Iteration {i} completed.\n'
                         f'{100*state.completion:.2f}% explored.\n'
                         f'Travelled {state.distance:.2f} meters.',
                         key='status', duration=np.inf)
        gui.save_screenshot(output_path(f'step{i:05d}.png'))

        np.savez(output_path(f'state{i:05d}.npz'), **state.save_dict)

        if 99.9/100 < state.completion:
            log.info('exploration complete')
            break


thread = run.thread
