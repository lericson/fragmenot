cimport numpy as np
from . cimport rtcore as rtc
from . cimport rtcore_ray as rtcr
from . cimport rtcore_scene as rtcs
from . cimport rtcore_geometry as rtcg
from .rtcore cimport Vertex, Triangle, Vec3f
from libc.stdlib cimport malloc, free

ctypedef Vec3f (*renderPixelFunc)(float x, float y,
                const Vec3f &vx, const Vec3f &vy, const Vec3f &vz,
                const Vec3f &p)

def run_triangles():
    pass

cdef unsigned int addCube(rtcs.RTCScene scene_i):

    cdef rtcg.RTCGeometry geom = rtcg.rtcNewGeometry(rtcs.rtcGetSceneDevice(scene_i), rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
    cdef unsigned int geomID = rtcg.rtcAttachGeometry(scene_i, geom)

    cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(geom, rtcg.RTC_BUFFER_TYPE_VERTEX, 0, rtcg.RTC_FORMAT_FLOAT3, 4*sizeof(float), 8)

    vertices[0].x = -1
    vertices[0].y = -1
    vertices[0].z = -1

    vertices[1].x = -1
    vertices[1].y = -1
    vertices[1].z = +1

    vertices[2].x = -1
    vertices[2].y = +1
    vertices[2].z = -1

    vertices[3].x = -1
    vertices[3].y = +1
    vertices[3].z = +1

    vertices[4].x = +1
    vertices[4].y = -1
    vertices[4].z = -1

    vertices[5].x = +1
    vertices[5].y = -1
    vertices[5].z = +1

    vertices[6].x = +1
    vertices[6].y = +1
    vertices[6].z = -1

    vertices[7].x = +1
    vertices[7].y = +1
    vertices[7].z = +1

    cdef Vec3f *colors = <Vec3f*> malloc(12*sizeof(Vec3f))

    cdef int tri = 0

    cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(geom, rtcg.RTC_BUFFER_TYPE_INDEX, 0, rtcg.RTC_FORMAT_UINT3, 3*sizeof(int), 12)

    # left side
    colors[tri].x = 1.0
    colors[tri].y = 0.0
    colors[tri].z = 0.0
    triangles[tri].v0 = 0
    triangles[tri].v1 = 2
    triangles[tri].v2 = 1
    tri += 1
    colors[tri].x = 1.0
    colors[tri].y = 0.0
    colors[tri].z = 0.0
    triangles[tri].v0 = 1
    triangles[tri].v1 = 2
    triangles[tri].v2 = 3
    tri += 1

    # right side
    colors[tri].x = 0.0
    colors[tri].y = 1.0
    colors[tri].z = 0.0
    triangles[tri].v0 = 4
    triangles[tri].v1 = 5
    triangles[tri].v2 = 6
    tri += 1
    colors[tri].x = 0.0
    colors[tri].y = 1.0
    colors[tri].z = 0.0
    triangles[tri].v0 = 5
    triangles[tri].v1 = 7
    triangles[tri].v2 = 6
    tri += 1

    # bottom side
    colors[tri].x = 0.5
    colors[tri].y = 0.5
    colors[tri].z = 0.5
    triangles[tri].v0 = 0
    triangles[tri].v1 = 1
    triangles[tri].v2 = 4
    tri += 1
    colors[tri].x = 0.5
    colors[tri].y = 0.5
    colors[tri].z = 0.5
    triangles[tri].v0 = 1
    triangles[tri].v1 = 5
    triangles[tri].v2 = 4
    tri += 1

    # top side
    colors[tri].x = 1.0
    colors[tri].y = 1.0
    colors[tri].z = 1.0
    triangles[tri].v0 = 2
    triangles[tri].v1 = 6
    triangles[tri].v2 = 3
    tri += 1
    colors[tri].x = 1.0
    colors[tri].y = 1.0
    colors[tri].z = 1.0
    triangles[tri].v0 = 3
    triangles[tri].v1 = 6
    triangles[tri].v2 = 7
    tri += 1

    # front side
    colors[tri].x = 0.0
    colors[tri].y = 0.0
    colors[tri].z = 1.0
    triangles[tri].v0 = 0
    triangles[tri].v1 = 4
    triangles[tri].v2 = 2
    tri += 1
    colors[tri].x = 0.0
    colors[tri].y = 0.0
    colors[tri].z = 1.0
    triangles[tri].v0 = 2
    triangles[tri].v1 = 4
    triangles[tri].v2 = 6
    tri += 1

    # back side
    colors[tri].x = 1.0
    colors[tri].y = 1.0
    colors[tri].z = 0.0
    triangles[tri].v0 = 1
    triangles[tri].v1 = 3
    triangles[tri].v2 = 5
    tri += 1
    colors[tri].x = 1.0
    colors[tri].y = 1.0
    colors[tri].z = 0.0
    triangles[tri].v0 = 3
    triangles[tri].v1 = 7
    triangles[tri].v2 = 5
    tri += 1

    rtcg.rtcCommitGeometry(geom)
    rtcg.rtcReleaseGeometry(geom)

    return geomID

cdef unsigned int addGroundPlane (rtcs.RTCScene scene_i):
    cdef rtcg.RTCGeometry geom = rtcg.rtcNewGeometry(rtcs.rtcGetSceneDevice(scene_i), rtcg.RTC_GEOMETRY_TYPE_TRIANGLE)
    cdef unsigned int geomID = rtcg.rtcAttachGeometry(scene_i, geom)

    cdef Vertex* vertices = <Vertex*> rtcg.rtcSetNewGeometryBuffer(geom, rtcg.RTC_BUFFER_TYPE_VERTEX, 0, rtcg.RTC_FORMAT_FLOAT3, 4*sizeof(float), 4)
    vertices[0].x = -10
    vertices[0].y = -2
    vertices[0].z = -10

    vertices[1].x = -10
    vertices[1].y = -2
    vertices[1].z = +10

    vertices[2].x = +10
    vertices[2].y = -2
    vertices[2].z = -10

    vertices[3].x = +10
    vertices[3].y = -2
    vertices[3].z = +10

    cdef Triangle* triangles = <Triangle*> rtcg.rtcSetNewGeometryBuffer(geom, rtcg.RTC_BUFFER_TYPE_INDEX, 0, rtcg.RTC_FORMAT_UINT3, 3*sizeof(int), 2)
    triangles[0].v0 = 0
    triangles[0].v1 = 2
    triangles[0].v2 = 1
    triangles[1].v0 = 1
    triangles[1].v1 = 2
    triangles[1].v2 = 3

    rtcg.rtcCommitGeometry(geom)
    rtcg.rtcReleaseGeometry(geom)

    return geomID
