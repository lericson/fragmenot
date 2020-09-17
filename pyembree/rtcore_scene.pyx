# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3, warn.unused_result=True, warn.unused_arg=True

cimport cython
cimport numpy as cnp
import numpy as np
import logging
import numbers
from . cimport rtcore as rtc
from . cimport rtcore_ray as rtcr


log = logging.getLogger('pyembree')

cdef void error_printer(void *ptr, const rtc.RTCError code, const char *_str):
    if _str:
        log.error('%s: %s', rtc.rtcErrorString(code), str(_str))


cdef class EmbreeScene:
    def __init__(self, rtc.EmbreeDevice device=None):
        if device is None:
            # We store the embree device inside EmbreeScene to avoid premature deletion
            self.device = rtc.EmbreeDevice()
            device = self.device
        rtc.rtcSetDeviceErrorFunction(device.device, error_printer, NULL)
        self.scene_i = rtcNewScene(device.device)
        rtcSetSceneFlags(self.scene_i, RTC_SCENE_FLAG_ROBUST)
        self.is_committed = 0

    # This was really poor API design.
    run = NotImplemented

    def intersects_ids(self,
                       const float [:, ::1] origins not None,
                       const float [:, ::1] directions not None,
                       max_dists = np.inf,
                       int [::1] out = None):

        if self.is_committed == 0:
            rtcCommitScene(self.scene_i)
            self.is_committed = 1

        cdef int n_rays, vo, vd, vf, vo_step, vd_step, vf_step, i, j, ij
        cdef float [:, ::1] tfars
        cdef rtcr.RTCRayHit   rayhit
        cdef rtcr.RTCRayHit16 rayhit16
        cdef RTCIntersectContext context

        n_rays = max(origins.shape[0], directions.shape[0])

        if isinstance(max_dists, numbers.Number):
            tfars = np.array([[max_dists]], dtype=np.float32)
        else:
            tfars = np.array(max_dists[:, None], dtype=np.float32)

        if out is None:
            out = np.empty(n_rays, dtype=np.int32)
        elif out.shape[0] != n_rays or out.shape[1] != 0:
            raise ValueError(f'given out array is not correct shape ({n_rays},)')

        rtcInitIntersectContext(&context)

        # Set stride to zero if we only have a single value. This stride trick
        # doesn't work for the last dimension (access is not multiplied by its
        # stride) which is why the tfars has a unit dimension at the end.
        if origins.shape[0] == 1:
            origins.strides[0] = 0

        if directions.shape[0] == 1:
            directions.strides[0] = 0

        if tfars.shape[0] == 1:
            tfars.strides[0] = 0

        cdef int chunk16_upto = 16*(n_rays//16)

        with cython.boundscheck(False), cython.wraparound(False), cython.nogil(True):

            for i in range(0, chunk16_upto, 16):

                for j in range(16):
                    ij = i+j
                    rayhit16.ray.org_x[j] = origins[ij, 0]
                    rayhit16.ray.org_y[j] = origins[ij, 1]
                    rayhit16.ray.org_z[j] = origins[ij, 2]
                    rayhit16.ray.dir_x[j] = directions[ij, 0]
                    rayhit16.ray.dir_y[j] = directions[ij, 1]
                    rayhit16.ray.dir_z[j] = directions[ij, 2]
                    rayhit16.ray.flags[j] = 0
                    rayhit16.ray.tnear[j] = 0.0
                    rayhit16.ray.tfar[j]  = tfars[ij, 0]
                    rayhit16.ray.mask[j]  = -1
                    rayhit16.ray.time[j]  = 0
                    rayhit16.hit.geomID[j]    = RTC_INVALID_GEOMETRY_ID
                    rayhit16.hit.primID[j]    = RTC_INVALID_GEOMETRY_ID
                    rayhit16.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID

                rtcIntersect16(rtcr.valid16, self.scene_i, &context, &rayhit16)

                for j in range(16):
                    out[i+j] = rayhit16.hit.primID[j]

            for i in range(chunk16_upto, n_rays):
                rayhit.ray.org_x = origins[i, 0]
                rayhit.ray.org_y = origins[i, 1]
                rayhit.ray.org_z = origins[i, 2]
                rayhit.ray.dir_x = directions[i, 0]
                rayhit.ray.dir_y = directions[i, 1]
                rayhit.ray.dir_z = directions[i, 2]
                rayhit.ray.flags = 0
                rayhit.ray.tnear = 0.0
                rayhit.ray.tfar = tfars[i, 0]
                rayhit.ray.mask = -1
                rayhit.ray.time = 0
                rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID
                rayhit.hit.primID    = RTC_INVALID_GEOMETRY_ID
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID

                rtcIntersect1(self.scene_i, &context, &rayhit)
                out[i] = rayhit.hit.primID

        return np.asarray(out)

    def __dealloc__(self):
        rtcReleaseScene(self.scene_i)
