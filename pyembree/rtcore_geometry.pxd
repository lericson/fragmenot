# rtcore_geometry wrapper

from .rtcore_ray cimport RTCRay, RTCRay4, RTCRay8, RTCRay16
from .rtcore_scene cimport RTCScene
from .rtcore cimport RTCDevice
cimport cython
cimport numpy as np

cdef extern from "embree3/rtcore_geometry.h":
    cdef unsigned int RTC_INVALID_GEOMETRY_ID

    cdef enum RTCBufferType:
        RTC_BUFFER_TYPE_INDEX
        RTC_BUFFER_TYPE_VERTEX
        RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE
        RTC_BUFFER_TYPE_NORMAL
        RTC_BUFFER_TYPE_TANGENT
        RTC_BUFFER_TYPE_NORMAL_DERIVATIVE

        RTC_BUFFER_TYPE_GRID

        RTC_BUFFER_TYPE_FACE
        RTC_BUFFER_TYPE_LEVEL
        RTC_BUFFER_TYPE_EDGE_CREASE_INDEX
        RTC_BUFFER_TYPE_EDGE_CREASE_WEIGHT
        RTC_BUFFER_TYPE_VERTEX_CREASE_INDEX
        RTC_BUFFER_TYPE_VERTEX_CREASE_WEIGHT
        RTC_BUFFER_TYPE_HOLE

        RTC_BUFFER_TYPE_FLAGS

    cdef enum RTCFormat:
        RTC_FORMAT_UNDEFINED

        # 8-bit unsigned integer
        RTC_FORMAT_UCHAR
        RTC_FORMAT_UCHAR2
        RTC_FORMAT_UCHAR3
        RTC_FORMAT_UCHAR4

        # 8-bit signed integer
        RTC_FORMAT_CHAR
        RTC_FORMAT_CHAR2
        RTC_FORMAT_CHAR3
        RTC_FORMAT_CHAR4

        # 16-bit unsigned integer
        RTC_FORMAT_USHORT
        RTC_FORMAT_USHORT2
        RTC_FORMAT_USHORT3
        RTC_FORMAT_USHORT4

        # 16-bit signed integer
        RTC_FORMAT_SHORT
        RTC_FORMAT_SHORT2
        RTC_FORMAT_SHORT3
        RTC_FORMAT_SHORT4

        # 32-bit unsigned integer
        RTC_FORMAT_UINT
        RTC_FORMAT_UINT2
        RTC_FORMAT_UINT3
        RTC_FORMAT_UINT4

        # 32-bit signed integer
        RTC_FORMAT_INT
        RTC_FORMAT_INT2
        RTC_FORMAT_INT3
        RTC_FORMAT_INT4

        # 64-bit unsigned integer
        RTC_FORMAT_ULLONG
        RTC_FORMAT_ULLONG2
        RTC_FORMAT_ULLONG3
        RTC_FORMAT_ULLONG4

        # 64-bit signed integer
        RTC_FORMAT_LLONG
        RTC_FORMAT_LLONG2
        RTC_FORMAT_LLONG3
        RTC_FORMAT_LLONG4

        # 32-bit float
        RTC_FORMAT_FLOAT
        RTC_FORMAT_FLOAT2
        RTC_FORMAT_FLOAT3
        RTC_FORMAT_FLOAT4
        RTC_FORMAT_FLOAT5
        RTC_FORMAT_FLOAT6
        RTC_FORMAT_FLOAT7
        RTC_FORMAT_FLOAT8
        RTC_FORMAT_FLOAT9
        RTC_FORMAT_FLOAT10
        RTC_FORMAT_FLOAT11
        RTC_FORMAT_FLOAT12
        RTC_FORMAT_FLOAT13
        RTC_FORMAT_FLOAT14
        RTC_FORMAT_FLOAT15
        RTC_FORMAT_FLOAT16

        # 32-bit float matrix (row-major order)
        RTC_FORMAT_FLOAT2X2_ROW_MAJOR
        RTC_FORMAT_FLOAT2X3_ROW_MAJOR
        RTC_FORMAT_FLOAT2X4_ROW_MAJOR
        RTC_FORMAT_FLOAT3X2_ROW_MAJOR
        RTC_FORMAT_FLOAT3X3_ROW_MAJOR
        RTC_FORMAT_FLOAT3X4_ROW_MAJOR
        RTC_FORMAT_FLOAT4X2_ROW_MAJOR
        RTC_FORMAT_FLOAT4X3_ROW_MAJOR
        RTC_FORMAT_FLOAT4X4_ROW_MAJOR

        # 32-bit float matrix (column-major order)
        RTC_FORMAT_FLOAT2X2_COLUMN_MAJOR
        RTC_FORMAT_FLOAT2X3_COLUMN_MAJOR
        RTC_FORMAT_FLOAT2X4_COLUMN_MAJOR
        RTC_FORMAT_FLOAT3X2_COLUMN_MAJOR
        RTC_FORMAT_FLOAT3X3_COLUMN_MAJOR
        RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR
        RTC_FORMAT_FLOAT4X2_COLUMN_MAJOR
        RTC_FORMAT_FLOAT4X3_COLUMN_MAJOR
        RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR

        # special 12-byte format for grids
        RTC_FORMAT_GRID

    cdef enum RTCGeometryType:
        RTC_GEOMETRY_TYPE_TRIANGLE  # triangle mesh
        RTC_GEOMETRY_TYPE_QUAD      # quad (triangle pair) mesh
        RTC_GEOMETRY_TYPE_GRID      # grid mesh

        RTC_GEOMETRY_TYPE_SUBDIVISION  # Catmull-Clark subdivision surface

        RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE  # Round (rounded cone like) linear curves 
        RTC_GEOMETRY_TYPE_FLAT_LINEAR_CURVE   # flat (ribbon-like) linear curves

        RTC_GEOMETRY_TYPE_ROUND_BEZIER_CURVE  # round (tube-like) Bezier curves
        RTC_GEOMETRY_TYPE_FLAT_BEZIER_CURVE   # flat (ribbon-like) Bezier curves
        RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_BEZIER_CURVE  # flat normal-oriented Bezier curves

        RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE  # round (tube-like) B-spline curves
        RTC_GEOMETRY_TYPE_FLAT_BSPLINE_CURVE   # flat (ribbon-like) B-spline curves
        RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_BSPLINE_CURVE  # flat normal-oriented B-spline curves

        RTC_GEOMETRY_TYPE_ROUND_HERMITE_CURVE  # round (tube-like) Hermite curves
        RTC_GEOMETRY_TYPE_FLAT_HERMITE_CURVE   # flat (ribbon-like) Hermite curves
        RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_HERMITE_CURVE  # flat normal-oriented Hermite curves

        RTC_GEOMETRY_TYPE_SPHERE_POINT
        RTC_GEOMETRY_TYPE_DISC_POINT
        RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT

        RTC_GEOMETRY_TYPE_ROUND_CATMULL_ROM_CURVE  # round (tube-like) Catmull-Rom curves
        RTC_GEOMETRY_TYPE_FLAT_CATMULL_ROM_CURVE   # flat (ribbon-like) Catmull-Rom curves
        RTC_GEOMETRY_TYPE_NORMAL_ORIENTED_CATMULL_ROM_CURVE  # flat normal-oriented Catmull-Rom curves

        RTC_GEOMETRY_TYPE_USER      # user-defined geometry
        RTC_GEOMETRY_TYPE_INSTANCE  # scene instance

    cdef enum RTCMatrixType:
        RTC_MATRIX_ROW_MAJOR
        RTC_MATRIX_COLUMN_MAJOR
        RTC_MATRIX_COLUMN_MAJOR_ALIGNED16

    cdef enum RTCGeometryFlags:
        RTC_GEOMETRY_STATIC
        RTC_GEOMETRY_DEFORMABLE
        RTC_GEOMETRY_DYNAMIC

    cdef struct RTCBounds:
        float lower_x, lower_y, lower_z, align0
        float upper_x, upper_y, upper_z, align1

    unsigned rtcNewTriangleMesh(RTCScene scene, RTCGeometryFlags flags, 
                                size_t numTriangles, size_t numVertices,
                                size_t numTimeSteps)

    ctypedef void* RTCGeometry

    void* rtcSetNewGeometryBuffer(RTCGeometry geometry, RTCBufferType type, unsigned int slot, RTCFormat format, size_t byteStride, size_t itemCount)

    RTCGeometry rtcNewGeometry(RTCDevice device, RTCGeometryType type)
    void rtcRetainGeometry(RTCGeometry geom)
    void rtcReleaseGeometry(RTCGeometry geom)
    void rtcCommitGeometry(RTCGeometry geom)
    void rtcEnableGeometry(RTCGeometry geom)
    void rtcDisableGeometry(RTCGeometry geom)

    unsigned int rtcAttachGeometry(RTCScene scene, RTCGeometry geom)
    void rtcAttachGeometryByID(RTCScene scene, RTCGeometry geom, unsigned int geomID)
    void rtcDetachGeometry(RTCScene scene, unsigned int geomID)

    void *rtcMapBuffer(RTCScene scene, unsigned geomID, RTCBufferType type)
    void rtcUnmapBuffer(RTCScene scene, unsigned geomID, RTCBufferType type)
