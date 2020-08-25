# rtcore_ray.pxd wrapper

cimport cython
cimport numpy as np

cdef extern from "embree3/rtcore_ray.h":
    """
    int valid16[16] = { -1, -1, -1, -1, -1, -1, -1, -1,
                        -1, -1, -1, -1, -1, -1, -1, -1  };
    """
    int valid16[];

    DEF RTC_MAX_INSTANCE_LEVEL_COUNT = 1

    # RTCORE_ALIGN(16)
    # This is for a *single* ray
    cdef struct RTCRay:
        # Ray data
        float org_x, org_y, org_z
        float tnear

        float dir_x, dir_y, dir_z
        float time

        float tfar
        unsigned int mask
        unsigned int id
        unsigned int flags

    cdef struct RTCHit:
        float Ng_x, Ng_y, Ng_z
        float u
        float v
        unsigned int primID
        unsigned int geomID
        unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT]

    cdef struct RTCRayHit:
        RTCRay ray
        RTCHit hit

    # This is for a packet of 4 rays
    cdef struct RTCRay4:
        # Ray data
        float org_x[4]
        float org_y[4]
        float org_z[4]
        float tnear[4]

        float dir_x[4]
        float dir_y[4]
        float dir_z[4]
        float time[4]

        float tfar[4]
        unsigned int mask[4]
        unsigned int id[4]
        unsigned int flags[4]

    cdef struct RTCHit4:
        # Hit data
        float Ng_x[4]
        float Ng_y[4]
        float Ng_z[4]
        float u[4]
        float v[4]
        int geomID[4]
        int primID[4]
        unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT][4]

    cdef struct RTCRayHit4:
        RTCRay4 ray
        RTCHit4 hit

    # This is for a packet of 8 rays
    cdef struct RTCRay8:
        # Ray data
        float org_x[8]
        float org_y[8]
        float org_z[8]
        float tnear[8]

        float dir_x[8]
        float dir_y[8]
        float dir_z[8]
        float time[8]

        float tfar[8]
        unsigned int mask[8]
        unsigned int id[8]
        unsigned int flags[8]

    cdef struct RTCHit8:
        # Hit data
        float Ng_x[8]
        float Ng_y[8]
        float Ng_z[8]
        float u[8]
        float v[8]
        int geomID[8]
        int primID[8]
        unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT][8]

    cdef struct RTCRayHit8:
        RTCRay8 ray
        RTCHit8 hit

    # This is for a packet of 16 rays
    cdef struct RTCRay16:
        # Ray data
        float org_x[16]
        float org_y[16]
        float org_z[16]
        float tnear[16]

        float dir_x[16]
        float dir_y[16]
        float dir_z[16]
        float time[16]

        float tfar[16]
        unsigned int mask[16]
        unsigned int id[16]
        unsigned int flags[16]

    cdef struct RTCHit16:
        # Hit data
        float Ng_x[16]
        float Ng_y[16]
        float Ng_z[16]
        float u[16]
        float v[16]
        int geomID[16]
        int primID[16]
        unsigned int instID[RTC_MAX_INSTANCE_LEVEL_COUNT][16]

    cdef struct RTCRayHit16:
        RTCRay16 ray
        RTCHit16 hit
