import logging

import octomap
import trimesh

import parame


cfg = parame.Module(__name__)
log = logging.getLogger(__name__)


@parame.configurable
def load(*,
         filename: cfg.param = 'resources/map.ot',
         resolution: cfg.param = 0.4):
    log.info(f'loading {filename}...')
    if filename.endswith('.ot'):
        octree = octomap.OcTree.read(bytes(filename, 'utf-8'))
    elif filename.endswith('.bt'):
        octree = octomap.OcTree(b"")
        octree.readBinary(bytes(filename, 'utf-8'))
    else:
        raise ValueError('do not know how to load this extension')

    log.debug('extracting pointcloud...')
    pts, _ = octree.extractPointCloud()

    log.info(f'marching cubing octree (res={resolution:.2f})...')
    mesh = trimesh.voxel.ops.points_to_marching_cubes(pts, pitch=resolution)
    log.info(f'done, created mesh of {mesh.faces.shape[0]} faces')

    return octree, mesh
