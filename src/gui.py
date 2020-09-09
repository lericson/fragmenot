#!/usr/bin/env python

import time
import logging
import threading
from colorsys import hsv_to_rgb

import trimesh
import trimesh.viewer
import numpy as np
from pyglet import gl, app
from pyglet.window import key
from networkx.utils import pairwise
from trimesh.path.entities import Line as LineEntity
from trimesh.transformations import translation_matrix

import parame


cfg = parame.Module(__name__)
log = logging.getLogger(__name__)

hsva_to_rgba = lambda h, s, v, a: np.array([*hsv_to_rgb(h, s, v), a])

color_envmesh       = hsva_to_rgba(0.70, 0.20, 0.80, 1.000)
color_envmesh_vis   = hsva_to_rgba(0.70, 1.00, 0.40, 1.000)
color_envmesh_aware = hsva_to_rgba(0.70, 1.00, 0.80, 1.000)
color_envmesh_seen  = hsva_to_rgba(0.33, 1.00, 0.80, 1.000)
color_envmesh_hl    = hsva_to_rgba(0.00, 1.00, 0.80, 1.000)

color_edge             = hsva_to_rgba(0.00, 0.00, 1.00, 0.375)
color_edge_skip        = hsva_to_rgba(0.00, 0.00, 1.00, 0.125)
color_edge_jump        = hsva_to_rgba(0.60, 0.40, 0.80, 0.125)
color_edge_hl          = hsva_to_rgba(0.22, 1.00, 0.80, 1.000)
color_edge_skip_hl     = hsva_to_rgba(0.22, 0.40, 0.80, 1.000)
color_edge_jump_hl     = hsva_to_rgba(0.50, 1.00, 0.80, 1.000)

color_trajectory_edge  = hsva_to_rgba(0.07, 1.00, 1.00, 0.900)

color_position_sphere  = hsva_to_rgba(0.07, 1.00, 1.00, 0.700)

# globals go brrrrrrr
_c = type('Context', (), {})()


@parame.configurable
def update_roadmap(roadmap, *,
                   show_jumps: cfg.param = False):
    path  = _c.roadmap_path
    lines = []
    verts = []

    for u, v, dd_uv in roadmap.edges.data():

        rs_uv = dd_uv['rs']

        if show_jumps:
            rs_uv = (rs_uv[0], rs_uv[-1])

        elif dd_uv['jump']:
            continue

        pts_uv = range(len(verts), len(verts) + len(rs_uv))
        line_uv = dd_uv['_line'] = LineEntity(points=pts_uv)
        lines.append(line_uv)
        verts.extend(rs_uv)

    reset_roadmap_edges(roadmap)

    with _c.viewer.lock:
        path.entities = np.array(lines)
        path.vertices = np.array(verts)


def reset_roadmap_edges(roadmap):
    for u, v, dd_uv in roadmap.edges.data():

        if '_line' not in dd_uv:
            continue

        line_uv = dd_uv['_line']
        line_uv.color = color_edge

        if dd_uv['jump']:
            line_uv.color = color_edge_jump
        elif dd_uv['skip']:
            line_uv.color = color_edge_skip


def hilight_roadmap_edges(roadmap, edges, reset=True):
    if reset:
        reset_roadmap_edges(roadmap)

    for u, v in edges:

        if (u, v) not in roadmap.edges:
            log.error(f'edge {u}-{v} does not exist')
            continue

        dd_uv = roadmap.edges[u, v]

        for w, w_ in pairwise(dd_uv['vs']):
            if (w, w_) in roadmap.edges:
                roadmap.edges[w, w_]['_line'].color = color_edge_hl
            else:
                log.error(f'edge {w}-{w_} does not exist')

        if dd_uv['jump']:
            clr = color_edge_jump_hl
        elif dd_uv['skip']:
            clr = color_edge_skip_hl
        else:
            clr = color_edge_hl

        dd_uv['_line'].color = clr

        assert dd_uv['_line'] in list(_c.roadmap_path.entities)


def update_vis_faces(*, visible=None, aware=None, seen=None, hilight=None):
    if visible is None: visible = np.zeros(_c.envmesh.faces.shape[0], dtype=bool)
    if aware is None:   aware   = np.zeros_like(visible)
    if seen is None:    seen    = np.zeros_like(visible)
    if hilight is None: hilight = np.zeros_like(visible)
    _c.layers['visible'][:]       = 255*color_envmesh
    _c.layers['visible'][visible] = 255*color_envmesh_vis
    _c.layers['visible'][aware]   = 255*color_envmesh_aware
    _c.layers['visible'][seen]    = 255*color_envmesh_seen
    _c.layers['visible'][hilight] = 255*color_envmesh_hl
    #_c.hl_faces_base_color = _c.envmesh.visual.face_colors.copy()


def hilight_vis_faces(faces):
    #if _c.hl_faces_base_color is not None:
    #    _c.envmesh.visual.face_colors[:] = _c.hl_faces_base_color
    _c.layers['visible'][faces] = 255*color_envmesh_hl


def update_face_hsv(*, layer, hues, saturation=1.0, value=1.0, alpha=1.0):
    _c.layers[layer][:] = 255*np.array([[*hsv_to_rgb(hue, saturation, value), alpha] for hue in hues])


def activate_layer(layer):
    data                = _c.layers[layer].copy()
    _c.layers[_c.layer] = _c.layers[_c.layer].copy()
    _c.layers[layer]    = _c.envmesh.visual.face_colors
    _c.layers[layer][:] = data
    _c.layer            = layer


def update_position(position):
    transform = translation_matrix(position - _c.sphere_position)
    _c.sphere_position = position
    _c.sphere.apply_transform(transform)
    if _c.follow:
        _c.viewer.view['ball']._n_pose[0:2, 3] = position[0:2]
        _c.scene.camera_transform[...]         = _c.viewer.view['ball'].pose


def update_trajectory(positions):
    v = range(len(positions))
    with _c.viewer.lock:
        path = _c.trajectory_path
        path.entities = [LineEntity(points=p) for p in zip(v[:-1], v[1:])]
        path.vertices = positions
        path.colors   = [color_trajectory_edge]*len(path.entities)


def wait_draw(timeout=1.0):
    _c.viewer._begin_draw.clear()
    if not _c.viewer._begin_draw.wait(timeout=timeout):
        log.warn('wait_draw timed out (begin draw)')
        return
    _c.viewer._end_draw.clear()
    if not _c.viewer._end_draw.wait(timeout=timeout):
        log.warn('wait_draw timed out (end draw)')


def save_screenshot(fn):
    log.info('saving screenshot: %s', fn)
    with _c.viewer._screenshot_lock:
        wait_draw(timeout=10.0)
        _c.viewer._screenshot_fn = fn
        wait_draw(timeout=10.0)


def show_message(*a, **k):
    _c.viewer.show_message(*a, **k)


def make_scene(*, mesh):
    "Create the base Scene"
    _c.hl_faces_base_color = None
    _c.roadmap_path    = trimesh.path.Path3D()
    _c.trajectory_path = trimesh.path.Path3D()
    _c.sphere          = trimesh.creation.icosphere(radius=0.25)
    _c.sphere.visual.vertex_colors[:] = 255*color_position_sphere
    _c.sphere_position = np.r_[0.0, 0.0, 0.0]

    scene = trimesh.Scene([mesh, _c.roadmap_path, _c.trajectory_path, _c.sphere])
    scene.set_camera(fov=[90, 90])
    return scene


class SceneViewer(trimesh.viewer.SceneViewer):
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
                self.save_image(self._screenshot_fn)
                self._screenshot_fn = None
            self._end_draw.set()

    def on_key_press(self, sym, mods):
        if sym == key.SPACE:
            layer_names = list(_c.layers)
            layer = (layer_names*2)[layer_names.index(_c.layer) + 1]
            activate_layer(layer)
            self.show_message(f'Switched to layer {layer}', key='layer', duration=1.0)
        elif sym == key.F:
            _c.follow = not _c.follow
            self.show_message(f'Following: {_c.follow}', key='follow')
        else:
            with self.lock:
                super().on_key_press(sym, mods)

    def _update_hud(self):
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

    viewer = SceneViewer(scene, resolution=resolution,
                         start_loop=False, **kw)
    #viewer.reset_view(flags=dict(cull=True, grid=True, wireframe=True))
    viewer.reset_view(flags=dict(perspective=perspective))

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


@parame.configurable
def init(envmesh, title=None, *,
         follow:    cfg.param = False,
         vsync:     cfg.param = True,
         minimized: cfg.param = False):

    _c.envmesh = envmesh
    _c.scene   = make_scene(mesh=envmesh)
    _c.viewer  = make_viewer(_c.scene, caption=title, vsync=vsync)
    _c.follow  = follow
    _c.layer   = 'visible'
    _c.layers  = {'visible': _c.envmesh.visual.face_colors,
                  'score': _c.envmesh.visual.face_colors.copy()}

    update_vis_faces()
    activate_layer('visible')

    if minimized:
        _c.viewer.minimize()

    return _c


run = app.run
close = app.exit