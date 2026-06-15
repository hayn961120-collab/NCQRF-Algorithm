# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _utils.pyx for details.

cimport numpy as cnp
from ..tree._tree cimport Node
from ._quantile cimport WeightedPQueueRecord
from ._typedefs cimport float32_t, float64_t, intp_t, uint8_t

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef float32_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (float32_t*)
    (intp_t*)
    (uint8_t*)
    (WeightedPQueueRecord*)
    (float64_t*)
    (float64_t**)
    (Node*)
    (Node**)


cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size)
