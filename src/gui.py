#!/usr/bin/env python

import time
import logging
import threading
import pathlib

import trimesh
import trimesh.viewer
import numpy as np
from networkx.utils import pairwise
from trimesh.path.entities import Line as LineEntity
from trimesh.transformations import translation_matrix

import parame
from utils import hsva_to_rgba


try:
    import pyglet.gl
    has_pyglet = True
except BaseException as e:
    from trimesh.exceptions import ExceptionModule
    pyglet = ExceptionModule(e)
    has_pyglet = False


cfg = parame.Module(__name__)
log = logging.getLogger(__name__)

f = lambda x: (255*x).astype(np.uint8)

color_envmesh          = f(hsva_to_rgba([0.00, 0.00, 0.90, 1.000]))
color_envmesh_vis      = f(hsva_to_rgba([0.70, 1.00, 0.40, 1.000]))
color_envmesh_aware    = f(hsva_to_rgba([0.70, 1.00, 0.80, 1.000]))
color_envmesh_seen     = f(hsva_to_rgba([0.33, 1.00, 0.80, 1.000]))
color_envmesh_hl       = f(hsva_to_rgba([0.00, 1.00, 0.80, 1.000]))

color_edge             = f(hsva_to_rgba([0.00, 0.00, 0.80, 0.200]))
color_edge_seen        = f(hsva_to_rgba([0.00, 0.00, 0.60, 0.125]))
color_edge_jump        = f(hsva_to_rgba([0.60, 1.00, 0.80, 0.125]))
color_edge_hl          = f(hsva_to_rgba([0.22, 0.90, 0.90, 0.990]))
color_edge_seen_hl     = f(hsva_to_rgba([0.22, 0.90, 0.70, 0.990]))
color_edge_jump_hl     = f(hsva_to_rgba([0.50, 1.00, 0.80, 0.990]))

color_edge_visited     = f(hsva_to_rgba([0.07, 1.00, 1.00, 0.990]))

color_position_sphere  = f(hsva_to_rgba([0.07, 1.00, 1.00, 0.700]))

# globals go brrrrrrr
_c = type('Context', (), {})()


@parame.configurable
def noop_when_headless(f, *,
                       headless: cfg.param = not has_pyglet):
    "Decorator replaces function with a no-op when in headless mode."
    if headless:
        return lambda *a, **k: None
    return f


@noop_when_headless
@parame.configurable
def update_roadmap(roadmap):

    path  = _c.roadmap_path
    lines = []
    verts = []

    for u, v, dd_uv in roadmap.edges.data():
        rs_uv   = dd_uv['rs']
        rs_uv   = [rs_uv[0], rs_uv[-1]]
        pts_uv  = range(len(verts), len(verts) + len(rs_uv))
        line_uv = dd_uv['_line'] = LineEntity(points=pts_uv)
        lines.append(line_uv)
        verts.extend(rs_uv)

    reset_roadmap_edges(roadmap)

    with _c.viewer.lock:
        path.entities = np.array(lines)
        path.vertices = np.array(verts)


@noop_when_headless
def reset_roadmap_edges(roadmap):
    for u, v, dd_uv in roadmap.edges.data():

        if '_line' not in dd_uv:
            continue

        line_uv = dd_uv['_line']
        line_uv.color = color_edge

        if dd_uv['jump']:
            line_uv.color = color_edge_jump
        elif dd_uv['visited']:
            line_uv.color = color_edge_visited
        elif dd_uv['seen']:
            line_uv.color = color_edge_seen


@noop_when_headless
def hilight_roadmap_edges(roadmap, edges, reset=True):
    if reset:
        reset_roadmap_edges(roadmap)

    for u, v in edges:

        if (u, v) not in roadmap.edges:
            log.error(f'edge {u}-{v} does not exist')
            continue

        dd_uv = roadmap.edges[u, v]

        for w, w_ in pairwise(dd_uv['vs']):
            if (w, w_) not in roadmap.edges:
                log.error(f'edge {w}-{w_} does not exist')
                continue
            dd_ww_ = roadmap.edges[w, w_]
            if dd_ww_['seen']:
                clr = color_edge_seen_hl
            else:
                clr = color_edge_hl
            dd_ww_['_line'].color = color_edge_hl

        if dd_uv['jump'] or 2 < len(dd_uv['vs']):
            clr = color_edge_jump_hl
        elif dd_uv['seen']:
            clr = color_edge_seen_hl
        else:
            clr = color_edge_hl

        dd_uv['_line'].color = clr

        assert dd_uv['_line'] in list(_c.roadmap_path.entities)


@noop_when_headless
def update_vis_faces(*, visible=None, aware=None, seen=None, hilight=None):
    if visible is None: visible = np.zeros(_c.envmesh.faces.shape[0], dtype=bool)
    if aware is None:   aware   = np.zeros_like(visible)
    if seen is None:    seen    = np.zeros_like(visible)
    if hilight is None: hilight = np.zeros_like(visible)
    _c.layers['visible'][:]       = color_envmesh
    _c.layers['visible'][visible] = color_envmesh_vis
    _c.layers['visible'][aware]   = color_envmesh_aware
    _c.layers['visible'][seen]    = color_envmesh_seen
    _c.layers['visible'][hilight] = color_envmesh_hl
    _update_if_active('visible')
    #_c.hl_faces_base_color = _c.envmesh.visual.face_colors.copy()


@noop_when_headless
def hilight_vis_faces(faces):
    #if _c.hl_faces_base_color is not None:
    #    _c.envmesh.visual.face_colors[:] = _c.hl_faces_base_color
    _c.layers['visible'][faces] = color_envmesh_hl
    _update_if_active('visible')


@noop_when_headless
def update_face_hsva(*, layer, h, s=1.0, v=1.0, a=1.0, faces=None):
    h, s, v, a = np.atleast_1d(h, s, v, a)
    hsva = np.empty_like(_c.layers[layer][faces, :], dtype=float)
    hsva[:, 0] = h
    hsva[:, 1] = s
    hsva[:, 2] = v
    hsva[:, 3] = a
    _c.layers[layer][faces, :] = (255.0*hsva_to_rgba(hsva)).astype(np.uint8)
    _update_if_active(layer)


@noop_when_headless
def update_face_color(color, *, layer, faces=None):
    colors = _c.layers[layer]
    colors[faces, :] = color
    _update_if_active(layer)


@noop_when_headless
def update_face_colormap(x, *, layer, alpha=1.0, faces=None):
    import turbo
    colors = _c.layers[layer]
    colors[faces, 0:3] = 255*turbo.interpolate(x)
    colors[faces,   3] = 255*alpha
    _update_if_active(layer)


@noop_when_headless
def _update_if_active(layer):
    if _c.layer == layer and _c.layers[layer] is not _c.envmesh.visual.face_colors:
        _c.envmesh.visual.face_colors[:] = _c.layers[layer]
        _c.layers[layer] = _c.envmesh.visual.face_colors


@noop_when_headless
def activate_layer(layer):
    data                = _c.layers[layer].copy()
    _c.layers[_c.layer] = _c.layers[_c.layer].copy()
    _c.layers[layer]    = _c.envmesh.visual.face_colors
    _c.layers[layer][:] = data
    _c.layer            = layer


@noop_when_headless
@parame.configurable
def reset_layers(*, default_layer: cfg.param = 'visible'):
    with _c.viewer.lock:
        current_layer = _c.envmesh.visual.face_colors
        _c.layer   = default_layer
        _c.layers  = {'visible': current_layer.copy(),
                      'score':   current_layer.copy()}
        _c.layers[default_layer] = current_layer
        _update_if_active(default_layer)


@noop_when_headless
def remove_faces(removed_faces):
    mesh = _c.envmesh
    removed_submesh, = mesh.submesh(np.nonzero(removed_faces))
    removed_submesh.visual.face_colors[:] = color_envmesh
    mesh.update_faces(~removed_faces)
    reset_layers()
    _c.scene.add_geometry(removed_submesh)


@noop_when_headless
def update_position(position):
    transform = translation_matrix(position - _c.sphere_position)
    _c.sphere_position = position
    _c.sphere.apply_transform(transform)
    if _c.follow:
        _c.viewer.view['ball']._n_pose[0:2, 3] = position[0:2]
        _c.scene.camera_transform[...]         = _c.viewer.view['ball'].pose


@noop_when_headless
def wait_draw(timeout=1.0):
    _c.viewer._begin_draw.clear()
    if not _c.viewer._begin_draw.wait(timeout=timeout):
        log.warn('wait_draw timed out (begin draw)')
        return
    _c.viewer._end_draw.clear()
    if not _c.viewer._end_draw.wait(timeout=timeout):
        log.warn('wait_draw timed out (end draw)')


@noop_when_headless
def save_screenshot(fn):
    log.info('saving screenshot: %s', fn)
    with _c.viewer._screenshot_lock:
        wait_draw(timeout=10.0)
        _c.viewer._screenshot_fn = fn
        wait_draw(timeout=10.0)


@noop_when_headless
def show_message(*a, **k):
    _c.viewer.show_message(*a, **k)


def make_scene(*, mesh):
    "Create the base Scene"
    _c.hl_faces_base_color = None
    _c.roadmap_path    = trimesh.path.Path3D()
    _c.trajectory_path = trimesh.path.Path3D()
    _c.sphere          = trimesh.creation.icosphere(radius=0.25)
    _c.sphere.visual.vertex_colors[:] = color_position_sphere
    _c.sphere_position = np.r_[0.0, 0.0, 0.0]

    scene = trimesh.Scene([mesh, _c.roadmap_path, _c.trajectory_path, _c.sphere])
    scene.set_camera(fov=[90, 90])
    return scene


BaseSceneViewer = trimesh.viewer.SceneViewer if has_pyglet else object
class SceneViewer(BaseSceneViewer):
    def init_gl(self):
        super().init_gl()
        self.lock = threading.Lock()
        self._begin_draw = threading.Event()
        self._end_draw   = threading.Event()
        self._screenshot_fn   = None
        self._screenshot_lock = threading.Lock()
        self._msgs = {}

    def on_draw(self):
        with self.lock:
            self._begin_draw.set()
            super().on_draw()
            if self._screenshot_fn is not None:
                self.save_image(str(self._screenshot_fn))
                self._screenshot_fn = None
            self._end_draw.set()

    def on_key_press(self, sym, mods):
        from pyglet.window import key
        if sym == key.SPACE:
            layer_names = list(_c.layers)
            layer = (layer_names*2)[layer_names.index(_c.layer) + 1]
            activate_layer(layer)
            self.show_message(f'Switched to layer {layer}', key='layer', duration=1.0)
        elif sym == key.F:
            _c.follow = not _c.follow
            self.show_message(f'Following: {_c.follow}', key='follow')
        elif sym == key.S:
            for i in range(1000):
                self._screenshot_fn = pathlib.Path(f'screenshot-{i:04d}.png')
                if not self._screenshot_fn.exists():
                    break
            self.show_message(f'Saving to {self._screenshot_fn}', key='screenshot')
        else:
            with self.lock:
                super().on_key_press(sym, mods)

    def _update_hud(self):
        if cfg.get('hide_messages', False):
            self._hud.text = ''
            return
        super()._update_hud()
        t_now = time.time()
        self._msgs = {key: (t, text) for key, (t, text) in self._msgs.items() if t_now < t}
        self._hud.text = '\n'.join([self._hud.text] + [text for _, text in self._msgs.values()])

    def show_message(self, text, *, duration=15.0, key=None):
        key = len(self._msgs) if key is None else key
        self._msgs[key] = (time.time() + duration, text)

    def invalidate_vertex_lists(self):
        self.vertex_list_hash.clear()


@parame.configurable
def make_viewer(scene, *,
                resolution:  cfg.param = np.r_[1280, 960],
                perspective: cfg.param = False,
                line_width:  cfg.param = 4,
                **kw):
    "Create a SceneViewer"

    # We _MUST_ pass a callback. Otherwise, the viewer class will not check for
    # new additions to the scene. It also means that new additions to the scene
    # will only be visible after a callback happens, which is once per draw
    # call, or once per callback_period.
    if 'callback' not in kw:
        kw['callback'] = lambda scene: None
        kw['callback_period'] = 1/8

    viewer = SceneViewer(scene, resolution=resolution, start_loop=False, **kw)
    #viewer.reset_view(flags=dict(cull=True, grid=True, wireframe=True))
    viewer.reset_view(flags=dict(perspective=perspective, cull=False))

    from pyglet import gl
    gl.glLineWidth(line_width)
    gl.glPointSize(line_width)

    log.info("""entering scene viewer interactive mode

keyboard shortcuts:

    w -- toggle wireframe
    c -- toggle backface culling
    g -- toggle grid
    q -- quit
""")

    return viewer


@noop_when_headless
@parame.configurable
def init(envmesh, title=None, *,
         follow:    cfg.param = False,
         vsync:     cfg.param = True,
         minimized: cfg.param = False):

    envmesh    = envmesh.copy()
    _c.envmesh = envmesh
    _c.scene   = make_scene(mesh=envmesh)
    _c.viewer  = make_viewer(_c.scene, caption=title, vsync=vsync)
    _c.follow  = follow

    reset_layers()

    if minimized:
        _c.viewer.minimize()

    return _c


if has_pyglet:
    run   = pyglet.app.run
    close = pyglet.app.exit
else:
    from trimesh.exceptions import closure
    exc = object.__getattribute__(pyglet, 'exc')
    run   = closure(exc)
    close = lambda: None
