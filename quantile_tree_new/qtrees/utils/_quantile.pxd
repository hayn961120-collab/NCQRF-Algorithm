# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _utils.pyx for details.

from ..tree._tree cimport Node
from ._typedefs cimport float32_t, float64_t, intp_t, uint8_t


# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    float64_t data
    float64_t weight

cdef class WeightedPQueue:
    cdef intp_t capacity
    cdef intp_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef intp_t size(self) noexcept nogil
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef int peek(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef float64_t get_weight_from_index(self, intp_t index) noexcept nogil
    cdef float64_t get_value_from_index(self, intp_t index) noexcept nogil


# =============================================================================
# WeightedQuantileCalculator data structure
# =============================================================================

cdef class WeightedQuantileCalculator:
    cdef intp_t initial_capacity
    cdef float64_t[::1] tau
    cdef intp_t ntaus
    cdef WeightedPQueue samples
    cdef float64_t total_weight
    cdef intp_t[::1] k
    cdef float64_t[::1] sum_w_0_k  # represents sum(weights[0:k]) = w[0] + w[1] + ... + w[k-1]
    cdef float64_t[::1] _quantiles
    cdef intp_t size(self) noexcept nogil
    cdef intp_t n_taus(self) noexcept nogil
    cdef int reset(self) except -1 nogil
    cdef int push(self, float64_t data, float64_t weight) except -1 nogil
    cdef int update_quantile_parameters_post_push(
        self, float64_t data, float64_t weight,
        float64_t[::1] original_quantile) noexcept nogil
    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil
    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil
    cdef int update_quantile_parameters_post_remove(
        self, float64_t data, float64_t weight,
        float64_t[::1] original_quantile) noexcept nogil 
    cdef float64_t[::1] get_quantile(self) noexcept nogil
    


# =============================================================================
# Weighted quantile-loss precomputation
# =============================================================================

cdef class WeightedQuantileLossPrecomputer:
    cdef intp_t n_samples
    cdef intp_t ntaus
    cdef bint initialized

    cdef object rank_by_sample_arr
    cdef object values_by_rank_arr
    cdef object weights_by_rank_arr
    cdef object tau_arr
    cdef object left_loss_arr
    cdef object right_loss_arr

    cdef intp_t[::1] rank_by_sample
    cdef float64_t[::1] values_by_rank
    cdef float64_t[::1] weights_by_rank
    cdef float64_t[::1] tau
    cdef float64_t[:, ::1] left_loss
    cdef float64_t[:, ::1] right_loss

    cpdef int init_node(self,
                        float64_t[::1] y,
                        float64_t[::1] sample_weight,
                        float64_t[::1] tau) except -1
    cpdef int compute_for_order(self, intp_t[::1] order) except -1
    cdef inline void _fenwick_add(self, float64_t[::1] bit,
                                  intp_t rank, float64_t value) noexcept nogil
    cdef inline float64_t _fenwick_prefix_sum(self, float64_t[::1] bit,
                                              intp_t rank) noexcept nogil
    cdef inline intp_t _fenwick_lower_bound(self, float64_t[::1] bit,
                                            float64_t target) noexcept nogil
    cdef inline float64_t _current_loss(self,
                                        float64_t[::1] bit_w,
                                        float64_t[::1] bit_y,
                                        float64_t total_w,
                                        float64_t total_y,
                                        float64_t tau) noexcept nogil
    cdef float64_t get_left_loss(self, intp_t tau_id, intp_t pos) noexcept nogil
    cdef float64_t get_right_loss(self, intp_t tau_id, intp_t pos) noexcept nogil
    cdef float64_t get_split_loss(self, intp_t tau_id, intp_t pos) noexcept nogil
