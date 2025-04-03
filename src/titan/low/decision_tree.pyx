cimport cython
from libc.stdlib cimport malloc, free
cimport numpy as np

cpdef best_split(np.ndarray[np.float64_t, ndim=2] arr):
    cdef int sample_count = arr.shape[0]
    print(f"sample_count: {sample_count}")
