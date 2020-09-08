# distutils: language=c++
# cython: language_level=3, warn.unused_result=True, warn.unused_arg=True

cdef extern from "<algorithm>" namespace "std" nogil:
    OutputIt set_difference[InputIt1, InputIt2, OutputIt](InputIt1 first1, InputIt1 last1,
                                                          InputIt2 first2, InputIt2 last2,
                                                          OutputIt d_first) except +
    Iter remove_if[Iter, UnaryPred](Iter first, Iter last, UnaryPred pred) except +

cdef extern from "<iterator>" namespace "std" nogil:
      ForwardIterator iter_next "std::next" [ForwardIterator] (ForwardIterator it) except +
