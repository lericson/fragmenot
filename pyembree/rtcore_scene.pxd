# rtcore_scene.pxd wrapper

cimport cython
cimport numpy as np
from . cimport rtcore as rtc
from . cimport rtcore_ray as rtcr

cdef extern from "embree3/rtcore_scene.h":


    cdef struct RTCIntersectContext:
        int flags
        void *filter
        unsigned int instID
        float distanceFac

    cdef int RTC_INVALID_GEOMETRY_ID

    cdef enum RTCSceneFlags:
        RTC_SCENE_FLAG_NONE
        RTC_SCENE_FLAG_DYNAMIC
        RTC_SCENE_FLAG_COMPACT
        RTC_SCENE_FLAG_ROBUST
        #RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION

    cdef enum RTCBuildQuality:
        RTC_BUILD_QUALITY_LOW
        RTC_BUILD_QUALITY_MEDIUM
        RTC_BUILD_QUALITY_HIGH
        RTC_BUILD_QUALITY_REFIT

    cdef enum RTCAlgorithmFlags:
        RTC_INTERSECT1
        RTC_INTERSECT4
        RTC_INTERSECT8
        RTC_INTERSECT16

    # ctypedef void* RTCDevice
    ctypedef void* RTCScene

    RTCScene rtcNewScene(rtc.RTCDevice device) nogil

    rtc.RTCDevice rtcGetSceneDevice(RTCScene scene) nogil

    void rtcSetSceneFlags(RTCScene scene, RTCSceneFlags flags) nogil
    void rtcCommitScene(RTCScene scene) nogil

    void rtcCommitThread(RTCScene scene, unsigned int threadID, unsigned int numThreads) nogil

    void rtcInitIntersectContext(RTCIntersectContext* context) nogil

    void rtcIntersect1(                    RTCScene scene, RTCIntersectContext* context, rtcr.RTCRayHit*   rayhit) nogil
    void rtcIntersect4( const void* valid, RTCScene scene, RTCIntersectContext* context, rtcr.RTCRayHit4*  rayhit) nogil
    void rtcIntersect8( const void* valid, RTCScene scene, RTCIntersectContext* context, rtcr.RTCRayHit8*  rayhit) nogil
    void rtcIntersect16(const void* valid, RTCScene scene, RTCIntersectContext* context, rtcr.RTCRayHit16* rayhit) nogil

    void rtcOccluded1(                    RTCScene scene, RTCIntersectContext* context, rtcr.RTCRay*   rayhit) nogil
    void rtcOccluded4( const void* valid, RTCScene scene, RTCIntersectContext* context, rtcr.RTCRay4*  rayhit) nogil
    void rtcOccluded8( const void* valid, RTCScene scene, RTCIntersectContext* context, rtcr.RTCRay8*  rayhit) nogil
    void rtcOccluded16(const void* valid, RTCScene scene, RTCIntersectContext* context, rtcr.RTCRay16* rayhit) nogil

    void rtcReleaseScene(RTCScene scene) nogil

cdef class EmbreeScene:
    cdef RTCScene scene_i
    # Optional device used if not given, it should be as input of EmbreeScene
    cdef public int is_committed
    cdef rtc.EmbreeDevice device

cdef enum rayQueryType:
    intersect,
    occluded,
    distance
