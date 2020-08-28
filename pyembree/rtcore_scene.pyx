cimport cython
cimport numpy as np
import numpy as np
import logging
import numbers
cimport rtcore as rtc
cimport rtcore_ray as rtcr


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

    def run(self, const float [:, ::1] vec_origins not None,
                  const float [:, ::1] vec_directions not None,
                  dists=None, query='INTERSECT', bint output = False):

        if self.is_committed == 0:
            rtcCommitScene(self.scene_i)
            self.is_committed = 1

        cdef int nv = vec_origins.shape[0]
        cdef int vo, vd, vd_step, i, j
        cdef int [::1] intersect_id
        cdef float [::1] tfar
        cdef rayQueryType query_type

        if query == 'INTERSECT':
            query_type = intersect
        elif query == 'OCCLUDED':
            query_type = occluded
        elif query == 'DISTANCE':
            query_type = distance
        else:
            raise ValueError("Embree ray query type %s not recognized." 
                "\nAccepted types are (INTERSECT,OCCLUDED,DISTANCE)" % (query))

        if dists is None:
            tfar = np.full(nv, 1e37, dtype=np.float32)
        elif isinstance(dists, numbers.Number):
            tfar = np.full(nv, dists, dtype=np.float32)
        else:
            tfar = dists

        cdef float [::1] u, v
        cdef float [:, ::1] Ng
        cdef int [::1] primID, geomID
        cdef int chunk16_upto = 16*(nv//16)

        if output:
            u  = np.empty(nv, dtype=np.float32)
            v  = np.empty(nv, dtype=np.float32)
            Ng = np.empty((nv, 3), dtype=np.float32)
            primID = np.empty(nv, dtype=np.int32)
            geomID = np.empty(nv, dtype=np.int32)
        else:
            intersect_id = np.empty(nv, dtype=np.int32)

        cdef rtcr.RTCRayHit   rayhit
        cdef rtcr.RTCRayHit16 rayhit16
        cdef RTCIntersectContext context

        rtcInitIntersectContext(&context)

        vd_step = 1
        # If vec_directions is 1 long, we won't be updating it.
        if vec_directions.shape[0] == 1:
            vd_step = 0

        with cython.boundscheck(False), cython.wraparound(False), cython.nogil(True):

            for i in range(0, chunk16_upto, 16):

                for j in range(16):
                    vo = i+j
                    vd = vd_step*vo
                    rayhit16.ray.org_x[j] = vec_origins[vo, 0]
                    rayhit16.ray.org_y[j] = vec_origins[vo, 1]
                    rayhit16.ray.org_z[j] = vec_origins[vo, 2]
                    rayhit16.ray.dir_x[j] = vec_directions[vd, 0]
                    rayhit16.ray.dir_y[j] = vec_directions[vd, 1]
                    rayhit16.ray.dir_z[j] = vec_directions[vd, 2]
                    rayhit16.ray.flags[j] = 0
                    rayhit16.ray.tnear[j] = 0.0
                    rayhit16.ray.tfar[j]  = tfar[vo]
                    rayhit16.ray.mask[j]  = -1
                    rayhit16.ray.time[j]  = 0
                    rayhit16.hit.geomID[j]    = RTC_INVALID_GEOMETRY_ID
                    rayhit16.hit.primID[j]    = RTC_INVALID_GEOMETRY_ID
                    rayhit16.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID

                if query_type == intersect or query_type == distance:
                    rtcIntersect16(rtcr.valid16, self.scene_i, &context, &rayhit16)

                    if not output:
                        if query_type == intersect:
                            for j in range(16):
                                intersect_id[i+j] = rayhit16.hit.primID[j]
                        else:
                            for j in range(16):
                                tfar[i+j] = rayhit16.ray.tfar[j]
                    else:
                        for j in range(16):
                            primID[i+j] = rayhit16.hit.primID[j]
                            geomID[i+j] = rayhit16.hit.geomID[j]
                            u[i+j] = rayhit16.hit.u[j]
                            v[i+j] = rayhit16.hit.v[j]
                            tfar[i+j] = rayhit16.ray.tfar[j]
                            Ng[i+j, 0] = rayhit16.hit.Ng_x[j]
                            Ng[i+j, 1] = rayhit16.hit.Ng_y[j]
                            Ng[i+j, 2] = rayhit16.hit.Ng_z[j]
                else:
                    rtcOccluded16(rtcr.valid16, self.scene_i, &context, &rayhit16.ray)
                    for j in range(16):
                        intersect_id[i+j] = rayhit16.hit.geomID[j]

            #for i in range(i, nv, 8):
            #for i in range(i, nv, 4):

            for i in range(chunk16_upto, nv):
                vo = i
                vd = vd_step*vo
                rayhit.ray.org_x = vec_origins[vo, 0]
                rayhit.ray.org_y = vec_origins[vo, 1]
                rayhit.ray.org_z = vec_origins[vo, 2]
                rayhit.ray.dir_x = vec_directions[vd, 0]
                rayhit.ray.dir_y = vec_directions[vd, 1]
                rayhit.ray.dir_z = vec_directions[vd, 2]
                rayhit.ray.flags = 0
                rayhit.ray.tnear = 0.0
                rayhit.ray.tfar = tfar[vo]
                rayhit.ray.mask = -1
                rayhit.ray.time = 0
                rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID
                rayhit.hit.primID    = RTC_INVALID_GEOMETRY_ID
                rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID

                if query_type == intersect or query_type == distance:
                    rtcIntersect1(self.scene_i, &context, &rayhit)

                    if not output:
                        if query_type == intersect:
                            intersect_id[i] = rayhit.hit.primID
                        else:
                            tfar[i] = rayhit.ray.tfar
                    else:
                        primID[i] = rayhit.hit.primID
                        geomID[i] = rayhit.hit.geomID
                        u[i] = rayhit.hit.u
                        v[i] = rayhit.hit.v
                        tfar[i] = rayhit.ray.tfar
                        Ng[i, 0] = rayhit.hit.Ng_x
                        Ng[i, 1] = rayhit.hit.Ng_y
                        Ng[i, 2] = rayhit.hit.Ng_z
                else:
                    rtcOccluded1(self.scene_i, &context, &rayhit.ray)
                    intersect_id[i] = rayhit.hit.geomID

        if output:
            return {'u':       np.asarray(u),
                    'v':       np.asarray(v), 
                    'Ng':      np.asarray(Ng), 
                    'tfar':    np.asarray(tfar), 
                    'primID':  np.asarray(primID), 
                    'geomID':  np.asarray(geomID)}
        else:
            if query_type == distance:
                return np.asarray(tfar)
            else:
                return np.asarray(intersect_id)

    def __dealloc__(self):
        rtcReleaseScene(self.scene_i)
