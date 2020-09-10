import logging

import logcolor

import gui
import world
import exploration


log = logging.getLogger(__name__)


def main():
    logcolor.basic_config(level=logging.DEBUG)
    logging.getLogger('trimesh').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    octree, mesh = world.load()

    if gui.cfg.get('headless', not gui.has_pyglet):
        exploration.run(octree=octree, mesh=mesh)

    else:
        gui.init(mesh, title=exploration.output_path())
        t = exploration.thread(octree=octree, mesh=mesh, daemon=True)
        t.start()
        gui.run()


if __name__ == '__main__':
    main()
