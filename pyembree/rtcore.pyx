# distutils: language=c++

embree_version = RTC_VERSION_MAJOR, RTC_VERSION_MINOR, RTC_VERSION_PATCH

error_code_desc = {
    RTC_ERROR_NONE:              "No error",
    RTC_ERROR_UNKNOWN:           "Unknown error",
    RTC_ERROR_INVALID_ARGUMENT:  "Invalid argument",
    RTC_ERROR_INVALID_OPERATION: "Invalid operation",
    RTC_ERROR_OUT_OF_MEMORY:     "Out of memory",
    RTC_ERROR_UNSUPPORTED_CPU:   "Unsupported CPU",
    RTC_ERROR_CANCELLED:         "Cancelled"
}

cdef str rtcErrorString(RTCError code):
    if code in error_code_desc:
        return error_code_desc[code]
    else:
        return "Unknown error code " + str(code)


cdef class EmbreeDevice:
    def __init__(self):
        self.device = rtcNewDevice(NULL)

    def __dealloc__(self):
        rtcReleaseDevice(self.device)

    def __repr__(self):
        return 'Embree version: {0}.{1}.{2}'.format(RTC_VERSION_MAJOR,
                                                    RTC_VERSION_MINOR,
                                                    RTC_VERSION_PATCH)
