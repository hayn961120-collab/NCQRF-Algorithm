# Authors: The scikit-learn developers
# Modifier: Beomjin Park
# SPDX-License-Identifier: BSD-3-Clause

from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.math cimport isnan

import numpy as np
cimport numpy as cnp
cnp.import_array()

# =============================================================================
# Helper functions
# =============================================================================

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError(f"could not allocate ({nelems} * {sizeof(p[0][0])}) bytes")

    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError(f"could not allocate {nbytes} bytes")

    p[0] = tmp
    return 0


cdef inline cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size):
    """Return copied data as 1D numpy array of intp's."""
    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> size
    return cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_INTP, data).copy()

def _any_isnan_axis0(const float32_t[:, :] X):
    """Same as np.any(np.isnan(X), axis=0)"""
    cdef:
        intp_t i, j
        intp_t n_samples = X.shape[0]
        intp_t n_features = X.shape[1]
        uint8_t[::1] isnan_out = np.zeros(X.shape[1], dtype=np.bool_)

    with nogil:
        for i in range(n_samples):
            for j in range(n_features):
                if isnan_out[j]:
                    continue
                if isnan(X[i, j]):
                    isnan_out[j] = True
                    break
    return np.asarray(isnan_out)