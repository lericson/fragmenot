import time
import logging
from os import makedirs, getpid, path, environ
from typing import Any, Optional
from dataclasses import dataclass

import yaml
import trimesh
import numpy as np
import networkx as nx
from networkx.utils import pairwise
from nptyping import NDArray

import gui
import prm
import parame
import prevision
import tree_search
from utils import threadable, format_duration


log = logging.getLogger(__name__)
cfg = parame.Module(__name__)


@dataclass(frozen=True)
class State():
    mesh: trimesh.Trimesh
    position: NDArray[(Any, 3)]
    seen_faces: NDArray[Any, bool]
    roadmap: nx.Graph
    node: Any = None
    trajectory: Optional[list] = None

    @classmethod
    @parame.configurable
    def new(cls, *, mesh, octree,
            start_position: cfg.param = 'random'):
        seen    = np.zeros(mesh.faces.shape[0], dtype=bool)
        start   = -1
        roadmap = prm.new(mesh=mesh, octree=octree)
        prm.insert_random(roadmap, start)
        seen   |= np.all([vis for _, _, vis in roadmap.edges(start, data='vis_faces')], axis=0)
        pos     = roadmap.nodes[start]['r']
        return cls(mesh, pos, seen, roadmap, start, trajectory=[pos])

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

    save_attrs = {'position', 'seen_faces', 'trajectory',
                  'distance', 'completion'}

    @property
    def save_dict(self):
        return {attr: getattr(self, attr) for attr in self.save_attrs}

    def update_gui(self, *, update_vis_faces=True):
        gui.update_roadmap(self.roadmap)
        gui.update_position(self.position)
        gui.update_trajectory(self.trajectory)
        if update_vis_faces:
            visible = self.roadmap.graph['vis_faces']
            gui.update_vis_faces(visible=visible, seen=self.seen_faces)


@threadable
def step(state):

    mesh    = state.mesh
    pos     = state.position
    seen    = state.seen_faces
    roadmap = state.roadmap
    node    = state.node
    traj    = state.trajectory

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

    prm.update_jumps(roadmap_, seen=seen, active={node})

    roadmap_local = prevision.subgraph(roadmap_, mesh=mesh, seen_faces=seen, seen_states=traj)

    gui.update_roadmap(roadmap_local)
    gui.wait_draw()

    path, vis_path = tree_search.plan_path(start=node, seen=seen,
                                           roadmap=roadmap_local)

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

    del node, pos, seen, traj, state

    state_ = State(mesh=mesh,
                   position=pos_,
                   seen_faces=seen_,
                   roadmap=roadmap_,
                   node=node_,
                   trajectory=traj_)

    state_.update_gui(update_vis_faces=False)
    gui.update_roadmap(roadmap_local)
    gui.hilight_roadmap_edges(roadmap_local, pairwise(path))
    gui.update_vis_faces(visible=roadmap_.graph['vis_faces'],
                         aware=roadmap_local.graph['vis_faces'],
                         seen=seen_,
                         hilight=vis_path & ~seen_)

    log.info('new state %s: %s', node_, pos_)
    log.info('explored %.2f%% of visible faces', 100*state_.completion)

    return state_


@parame.configurable
def output_path(*args, create=True,
                runs_dir:   cfg.param = './var/runs',
                output_dir: cfg.param = f'run_{getpid()}'):
    output_path = path.join(runs_dir, output_dir)
    if create:
        makedirs(output_path, exist_ok=True)
    return path.join(output_path, *args)


@threadable
@parame.configurable
def run(*, octree, mesh, state=None,
        num_steps:        cfg.param = 5000,
        percent_complete: cfg.param = 100.0,
        close_on_finish:  cfg.param = True):

    with open(output_path('environ.yaml'), 'w') as f:
        yaml.dump(dict(environ=dict(environ), parame=parame._file_cfg), stream=f)
        #f.write('Environment:\n')
        #f.write(''.join(f'{k}: {v}\n' for k, v in environ.items()))
        #f.write('parame file configuration:\n')
        #f.write(''.join(f'{k}: {v}\n' for k, v in parame._file_cfg))

    np.savez_compressed(output_path('mesh.npz'),
                        faces=mesh.faces,
                        vertices=mesh.vertices,
                        area_faces=mesh.area_faces)

    gui.activate_layer('visible')

    t0 = time.time()

    for i in range(num_steps):

        log.info('step %d begins', i)

        if state is None:
            state = State.new(octree=octree, mesh=mesh)
            state.update_gui()

        else:
            state = step(state)

        gui.show_message(f'Step {i} completed.\n'
                         f'{100*state.completion:.2f}% explored.\n'
                         f'Travelled {state.distance:.2f} meters.\n'
                         f'{len(state.roadmap.edges())} edges in roadmap.\n'
                         f'{format_duration(time.time()-t0)} on the clock.',
                         key='status', duration=np.inf)
        gui.save_screenshot(output_path(f'step{i:05d}.png'))
        np.savez(output_path(f'state{i:05d}.npz'), **state.save_dict)

        log.info('step %d ends\n', i)

        if not (state.completion < percent_complete/100):
            log.info('exploration complete!')
            log.info('link: file://{path.abspath(output_path())}')
            break

    if close_on_finish:
        gui.close()


thread = run.thread
