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
from utils import threadable, graph2ndarrays, format_duration, endswith_cycle


log = logging.getLogger(__name__)
cfg = parame.Module(__name__)


@dataclass(frozen=True)
class State():
    mesh:       trimesh.Trimesh
    seen_faces: NDArray[Any, bool]
    roadmap:    nx.Graph
    node:       Any
    sequence:   Optional[list]

    @classmethod
    @parame.configurable
    def new(cls, *, mesh, octree, start=-1,
            remove_faces:   cfg.param = True,
            start_position: cfg.param = 'random'):
        rd = prm.new(mesh=mesh, octree=octree)
        if remove_faces:
            prm.remove_invisible_faces(rd)
            mesh.update_faces(~rd.graph['removed_faces'])
            gui.remove_faces(rd.graph['removed_faces'])
        prm.insert(rd, start, position=start_position)

        seen = np.all([vis_uv for _, _, vis_uv in rd.edges(start, data='vis_faces')], axis=0)

        return cls(mesh, seen, rd, start, sequence=[start])

    @property
    def trajectory(self) -> list:
        return [self.roadmap.nodes[u]['r'] for u in self.sequence]

    @property
    def position(self) -> NDArray[3, np.float]:
        return self.trajectory[-1]

    @property
    def distance(self) -> float:
        if len(self.sequence) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(self.trajectory, axis=0), axis=1))

    @property
    def completion(self) -> float:
        visible    = self.roadmap.graph['vis_faces']
        seen       = self.seen_faces
        area_faces = self.mesh.area_faces
        return np.sum(area_faces[seen]) / np.sum(area_faces[visible])

    save_attrs = {'position', 'seen_faces', 'trajectory', 'sequence',
                  'distance', 'completion'}

    @property
    def save_dict(self) -> dict:
        return {attr: getattr(self, attr) for attr in self.save_attrs}

    def update_gui(self, *, update_vis_faces=True):
        gui.update_roadmap(self.roadmap)
        gui.update_position(self.position)
        if update_vis_faces:
            visible = self.roadmap.graph['vis_faces']
            gui.update_vis_faces(visible=visible, seen=self.seen_faces)


@threadable
@parame.configurable
def step(state, *,
         jump_edges: cfg.param = True):

    mesh    = state.mesh
    seen    = state.seen_faces
    roadmap = state.roadmap
    node    = state.node
    seq     = state.sequence

    assert seq[-1] == node

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

    if jump_edges:
        prm.update_jumps(roadmap_, seen=seen, active={node})

    roadmap_local = prevision.subgraph(roadmap_, mesh=mesh, seen_faces=seen, seen_states=seq)

    gui.update_roadmap(roadmap_local)
    gui.wait_draw()

    path, vis_path = tree_search.plan_path(start=node, seen=seen,
                                           roadmap=roadmap_local)

    log.info('chose path: %s', path)

    # node_ is the next node in the chosen path
    node_ = path[1]

    # Construct next state (hence the _ suffixes)
    seen_      = seen | roadmap_.edges[node, node_]['vis_faces']
    #seen_      = data_v['vis_faces'].copy()
    #roadmap_   = rrt.new(pos_) # NOTE should set root d = data_v['d']
    #roadmap_   = rrt.reroot(roadmap, node_)
    #roadmap_   = rrt.extract_path(roadmap, path[1:])
    #roadmap_   = rrt.new(pos_, d=data_v['d']) # NOTE should set root d = data_v['d']

    vs = roadmap_.edges[node, node_]['vs']
    vs = vs if vs[0] == node else vs[::-1]
    assert vs[0] == node and vs[-1] == node_

    # This makes it so taking a jump edge only moves one step along it.
    vs    = vs[:2]
    node_ = vs[-1]

    seq_ = seq + vs[1:]

    for u, v in pairwise(vs):
        roadmap_local.edges[u, v]['visited'] = True
        roadmap_.edges[u, v]['visited'] = True

    del node, seen, seq, state

    state_ = State(mesh=mesh,
                   seen_faces=seen_,
                   roadmap=roadmap_,
                   node=node_,
                   sequence=seq_)

    state_.update_gui(update_vis_faces=False)
    gui.update_roadmap(roadmap_local)
    gui.hilight_roadmap_edges(roadmap_local, pairwise(path[0:]))
    gui.update_vis_faces(visible=roadmap_.graph['vis_faces'],
                         aware=roadmap_local.graph['vis_faces'],
                         seen=seen_,
                         hilight=vis_path & ~seen_)

    log.info('new state %s: %s', state_.node, state_.position)
    log.info('completed %.2f%%', 100*state_.completion)
    log.info('travelled %.2f meters', state_.distance)

    return state_


@parame.configurable
def output_path(*args, create=True,
                runs_dir:   cfg.param = './var/runs',
                output_dir: cfg.param = 'run_{run_id}'):
    output_path = path.join(runs_dir, output_dir.format(run_id=getpid()))
    if create:
        makedirs(output_path, exist_ok=True)
    return path.join(output_path, *args)


assert not path.exists(output_path(create=False))


@threadable
@parame.configurable
def run(*, octree, mesh, state=None,
        num_steps:        cfg.param = 2000,
        percent_complete: cfg.param = 100.0,
        close_on_finish:  cfg.param = True):

    with open(output_path('environ.yaml'), 'w') as f:
        yaml.dump(dict(environ=dict(environ), parame=parame._file_cfg), stream=f)
        #f.write('Environment:\n')
        #f.write(''.join(f'{k}: {v}\n' for k, v in environ.items()))
        #f.write('parame file configuration:\n')
        #f.write(''.join(f'{k}: {v}\n' for k, v in parame._file_cfg))

    log.debug('creating initial state')
    if state is None:
        state = State.new(octree=octree, mesh=mesh)
        state.update_gui()

    log.debug('saving mesh to disk')
    np.savez_compressed(output_path('mesh.npz'),
                        faces=mesh.faces,
                        vertices=mesh.vertices)

    #log.debug('saving roadmap to disk')
    #np.savez_compressed(output_path('roadmap.npz'),
    #                    **graph2ndarrays(state.roadmap))

    t0 = time.time()

    for i in range(num_steps):

        log.info('step %d begins', i)

        gui.show_message(f'Step {i} begins.\n'
                         f'{100*state.completion:.2f}% explored.\n'
                         f'Travelled {state.distance:.2f} meters.\n'
                         f'{len(state.roadmap.edges())} edges in roadmap.\n'
                         f'{format_duration(time.time()-t0)} on the clock.',
                         key='status', duration=np.inf)
        gui.save_screenshot(output_path(f'step{i:05d}.png'))
        np.savez(output_path(f'state{i:05d}.npz'), **state.save_dict)

        if not (state.completion < percent_complete/100):
            log.info('exploration complete!')
            break

        if any(endswith_cycle(state.sequence, n=3, k=k) for k in {1,2,3}):
            log.error('exploration stuck in a cycle, aborting')
            break

        try:
            state = step(state)
        except tree_search.NoPlanFoundError:
            log.error('unable to find any path, aborting')
            break

        log.info('step %d ends\n', i)

    else:
        log.error('exploration did not finish in %d steps', num_steps)

    log.info('elapsed time: %s', format_duration(time.time()-t0))
    log.info('link: %s', f'file://{path.abspath(output_path())}')

    if close_on_finish:
        gui.close()


thread = run.thread
