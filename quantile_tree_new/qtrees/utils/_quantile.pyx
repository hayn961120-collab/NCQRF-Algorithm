# Authors: The scikit-learn developers
# Modifier: Beomjin Park
# SPDX-License-Identifier: BSD-3-Clause

from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.math cimport isnan

from ._utils cimport safe_realloc

from libc.string cimport memcpy
from libc.string cimport memset
from libc.string cimport memmove

import numpy as np
cimport numpy as cnp
cnp.import_array()


# =============================================================================
# WeightedPQueue data structure
# =============================================================================

cdef class WeightedPQueue:
    """A priority queue class, always sorted in increasing order.

    Attributes
    ----------
    capacity : intp_t
        The capacity of the priority queue.

    array_ptr : intp_t
        The water mark of the priority queue; the priority queue grows from
        left to right in the array ``array_``. ``array_ptr`` is always
        less than ``capacity``.

    array_ : WeightedPQueueRecord*
        The array of priority queue records. The minimum element is on the
        left at index 0, and the maximum element is on the right at index
        ``array_ptr-1``.
    """

    def __cinit__(self, intp_t capacity):
        self.capacity = capacity
        self.array_ptr = 0
        safe_realloc(&self.array_, capacity)

    def __dealloc__(self):
        free(self.array_)

    cdef int reset(self) except -1 nogil:
        """Reset the WeightedPQueue to its state at construction

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.array_ptr = 0
        return 0

    cdef bint is_empty(self) noexcept nogil:
        return self.array_ptr <= 0

    cdef intp_t size(self) noexcept nogil:
        return self.array_ptr

    cdef int push(self, float64_t data, float64_t weight) except -1 nogil:
        """Push record on the array.

        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef intp_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = NULL
        cdef intp_t i
        cdef intp_t lo
        cdef intp_t hi
        cdef intp_t mid
        cdef intp_t pos

        # Resize if capacity not sufficient
        if array_ptr >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.array_, self.capacity)

        array = self.array_

        # Binary search insertion point; keep sorted order while avoiding
        # repeated element-wise bubble swaps.
        lo = 0
        hi = array_ptr

        while lo < hi:
            mid = (lo + hi) >> 1
            if array[mid].data < data:
                lo = mid + 1
            else:
                hi = mid
        pos = lo

        if pos < array_ptr:
            memmove(&array[pos + 1], &array[pos],
                    (array_ptr - pos) * sizeof(WeightedPQueueRecord))

        array[pos].data = data
        array[pos].weight = weight

        self.array_ptr = array_ptr + 1
        return 0

    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil:
        """Remove a specific value/weight record from the array.
        Returns 0 if successful, -1 if record not found."""
        cdef intp_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = self.array_
        cdef intp_t idx_to_remove = -1
        cdef intp_t i

        if array_ptr <= 0:
            return -1

        # find element to remove
        for i in range(array_ptr):
            if array[i].data == data and array[i].weight == weight:
                idx_to_remove = i
                break

        if idx_to_remove == -1:
            return -1

        # Shift tail with C-level memmove rather than a Cython loop.
        if idx_to_remove < array_ptr - 1:
            memmove(&array[idx_to_remove], &array[idx_to_remove + 1],
                    (array_ptr - idx_to_remove - 1) * sizeof(WeightedPQueueRecord))

        self.array_ptr = array_ptr - 1
        return 0

    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil:
        """Remove the top (minimum) element from array.
        Returns 0 if successful, -1 if nothing to remove."""
        cdef intp_t array_ptr = self.array_ptr
        cdef WeightedPQueueRecord* array = self.array_
        cdef intp_t i

        if array_ptr <= 0:
            return -1

        data[0] = array[0].data
        weight[0] = array[0].weight

        # Shift tail with C-level memmove rather than a Cython loop.
        if array_ptr > 1:
            memmove(&array[0], &array[1],
                    (array_ptr - 1) * sizeof(WeightedPQueueRecord))

        self.array_ptr = array_ptr - 1
        return 0

    cdef int peek(self, float64_t* data, float64_t* weight) noexcept nogil:
        """Write the top element from array to a pointer.
        Returns 0 if successful, -1 if nothing to write."""
        cdef WeightedPQueueRecord* array = self.array_
        if self.array_ptr <= 0:
            return -1
        # Take first value
        data[0] = array[0].data
        weight[0] = array[0].weight
        return 0

    cdef float64_t get_weight_from_index(self, intp_t index) noexcept nogil:
        """Given an index between [0,self.current_capacity], access
        the appropriate heap and return the requested weight"""
        cdef WeightedPQueueRecord* array = self.array_

        # get weight at index
        return array[index].weight

    cdef float64_t get_value_from_index(self, intp_t index) noexcept nogil:
        """Given an index between [0,self.current_capacity], access
        the appropriate heap and return the requested value"""
        cdef WeightedPQueueRecord* array = self.array_

        # get value at index
        return array[index].data

# =============================================================================
# WeightedQuantileCalculator data structure
# =============================================================================

cdef class WeightedQuantileCalculator:
    """
    A class to handle calculation of the weighted quantile from streams of
    data. This generalizes the weighted median to other quantile thresholds.

    Attributes
    ----------
    initial_capacity : intp_t
        The initial capacity of the WeightedQuantileCalculator.

    samples : WeightedPQueue
        Holds the samples (values and their weights).

    total_weight : float64_t
        The sum of weights of all samples.

    k : intp_t
        The index used to calculate the quantile.

    sum_w_0_k : float64_t
        The cumulative sum of weights from samples[0:k].
    """

    def __cinit__(self, intp_t initial_capacity, float64_t[::1] tau):
        self.initial_capacity = initial_capacity
        self.tau = tau
        self.ntaus = tau.size
        self.samples = WeightedPQueue(initial_capacity)
        self.total_weight = 0
        self.k = np.zeros(self.ntaus, dtype = np.intp)
        self.sum_w_0_k = np.zeros(self.ntaus, dtype = np.float64)
        self._quantiles = np.zeros(self.ntaus, dtype = np.float64)

    cdef intp_t size(self) noexcept nogil:
        """Return the number of samples in the
        WeightedQuantileCalculator"""
        return self.samples.size()

    cdef intp_t n_taus(self) noexcept nogil:
        return self.ntaus

    cdef int reset(self) except -1 nogil:
        """Reset the calculator to its initial state."""
        self.samples.reset()
        self.total_weight = 0
        self.k[:] = 0
        self.sum_w_0_k[:] = 0.0
        return 0

    cdef int push(self, float64_t data, float64_t weight) except -1 nogil:
        """Push a value and its associated weight to the WeightedQuantileCalculator
        
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef int return_value 
        cdef float64_t[::1] original_quantile

        if self.size() != 0:
            original_quantile = self.get_quantile()

        return_value = self.samples.push(data, weight)
        self.update_quantile_parameters_post_push(data, weight, 
                                                  original_quantile)
        return return_value
   
    cdef int update_quantile_parameters_post_push(
        self, float64_t data, float64_t weight, 
        float64_t[::1] original_quantile,
        ) noexcept nogil:
        
        cdef intp_t j = 0
        cdef intp_t j_0 = 0

        if self.size() == 1:
            self.k[:] = 1
            self.total_weight = weight
            self.sum_w_0_k[:] = self.total_weight
            return 0
        
        self.total_weight += weight

        while(j >= 0 and j < self.n_taus() and (original_quantile[j] <= data)):
            j += 1            
            j_0 += 1
        
        while(j < self.n_taus()):
            self.k[j] += 1
            self.sum_w_0_k[j] += weight
            while(self.k[j] > 1 and ((self.sum_w_0_k[j] - self.samples.get_weight_from_index(self.k[j]-1))
                                    >= self.total_weight * self.tau[j])):
                self.k[j] -= 1
                self.sum_w_0_k[j] -= self.samples.get_weight_from_index(self.k[j])
            j += 1
        

        j = j_0
        while(j > 0):
            while(self.k[j-1] < self.samples.size() and 
                    (self.sum_w_0_k[j-1] < self.total_weight * self.tau[j-1])):
                self.k[j-1] += 1
                self.sum_w_0_k[j-1] += self.samples.get_weight_from_index(self.k[j-1]-1)
            j -= 1

        return 0
        

    cdef int remove(self, float64_t data, float64_t weight) noexcept nogil:
        
        cdef int return_value
        cdef float64_t[::1] original_quantile

        if self.size() != 0:
            original_quantile = self.get_quantile()

        return_value = self.samples.remove(data, weight)
        self.update_quantile_parameters_post_remove(data, weight,
                                                    original_quantile)
        return return_value


    cdef int pop(self, float64_t* data, float64_t* weight) noexcept nogil:
        """Pop a value from the QuantileHeap, starting from the
        left and moving to the right.
        """
        cdef int return_value
        cdef float64_t[::1] original_quantile

        if self.size() != 0:
            original_quantile = self.get_quantile()

        # no elements to pop
        if self.samples.size() == 0:
            return -1

        return_value = self.samples.pop(data, weight)
        self.update_quantile_parameters_post_remove(data[0],
                                                  weight[0],
                                                  original_quantile)
        return return_value


    cdef int update_quantile_parameters_post_remove(
        self, float64_t data, float64_t weight,
        float64_t[::1] original_quantile
    ) noexcept nogil:

        cdef intp_t j = 0
        cdef intp_t j_0 = 0

        if self.samples.size() == 0:
            self.k[:] = 0
            self.total_weight = 0
            self.sum_w_0_k[:] = 0.0
            return 0

        if self.samples.size() == 1:
            self.k[:] = 1
            self.total_weight -= weight
            self.sum_w_0_k[:] = self.total_weight
            return 0

        self.total_weight -= weight

        while(j >= 0 and j < self.n_taus() and (original_quantile[j] <= data)):
            j += 1            
            j_0 += 1
            
        while(j < self.n_taus()):
            self.k[j] -= 1
            self.sum_w_0_k[j] -= weight
            while(self.k[j] < self.samples.size() and 
                    (self.sum_w_0_k[j] < self.total_weight * self.tau[j])):
                self.k[j] += 1
                self.sum_w_0_k[j] += self.samples.get_weight_from_index(self.k[j]-1)
            j += 1

        j = j_0
        while(j > 0):
            while(self.k[j-1] > 1 and ((self.sum_w_0_k[j-1] - self.samples.get_weight_from_index(self.k[j-1]-1)) 
                                    >= self.total_weight * self.tau[j-1])):
                self.k[j-1] -= 1
                self.sum_w_0_k[j-1] -= self.samples.get_weight_from_index(self.k[j-1])
            j -= 1
        return 0


    cdef float64_t[::1] get_quantile(self) noexcept nogil:
        """
        Calculate the weighted quantile for the given proportion p.

        Returns
        -------
        quantiles : float64_t[::1]
            The weighted quantile values.
        """
        cdef intp_t i = 0

        for i in range(self.n_taus()):
            self._quantiles[i] = self.samples.get_value_from_index(self.k[i] - 1)
        return self._quantiles


# =============================================================================
# Weighted quantile-loss precomputation
# =============================================================================

cdef class WeightedQuantileLossPrecomputer:
    """Precompute prefix/suffix weighted quantile losses for split scans.

    Usage pattern:
    1. init_node(y_node, sample_weight_node, tau) once for the current node.
    2. compute_for_order(order) for each feature-specific sorted order.
    3. read left/right losses at a split position pos.

    The split after pos samples has left=order[:pos] and right=order[pos:].
    """

    def __cinit__(self):
        self.n_samples = 0
        self.ntaus = 0
        self.initialized = False

    cpdef int init_node(self,
                        float64_t[::1] y,
                        float64_t[::1] sample_weight,
                        float64_t[::1] tau) except -1:
        cdef intp_t n = y.shape[0]
        cdef intp_t i, r
        cdef cnp.ndarray order_np
        cdef cnp.ndarray rank_np
        cdef cnp.ndarray values_np
        cdef cnp.ndarray weights_np
        cdef cnp.ndarray tau_np
        cdef intp_t[::1] order

        if sample_weight.shape[0] != n:
            raise ValueError("sample_weight must have the same length as y")
        if tau.shape[0] == 0:
            raise ValueError("tau must contain at least one quantile level")

        self.n_samples = n
        self.ntaus = tau.shape[0]

        order_np = np.asarray(y).argsort(kind="mergesort").astype(np.intp, copy=False)
        rank_np = np.empty(n, dtype=np.intp)
        values_np = np.empty(n, dtype=np.float64)
        weights_np = np.empty(n, dtype=np.float64)
        tau_np = np.asarray(tau, dtype=np.float64).copy()

        order = order_np
        self.rank_by_sample_arr = rank_np
        self.values_by_rank_arr = values_np
        self.weights_by_rank_arr = weights_np
        self.tau_arr = tau_np

        self.rank_by_sample = rank_np
        self.values_by_rank = values_np
        self.weights_by_rank = weights_np
        self.tau = tau_np

        for r in range(n):
            i = order[r]
            self.rank_by_sample[i] = r
            self.values_by_rank[r] = y[i]
            self.weights_by_rank[r] = sample_weight[i]

        self.left_loss_arr = np.zeros((self.ntaus, n + 1), dtype=np.float64)
        self.right_loss_arr = np.zeros((self.ntaus, n + 1), dtype=np.float64)
        self.left_loss = self.left_loss_arr
        self.right_loss = self.right_loss_arr
        self.initialized = True
        return 0

    cpdef int compute_for_order(self, intp_t[::1] order) except -1:
        cdef intp_t n = self.n_samples
        cdef intp_t k, pos, sample, rank
        cdef float64_t total_w, total_y, loss
        cdef cnp.ndarray bit_w_np
        cdef cnp.ndarray bit_y_np
        cdef float64_t[::1] bit_w
        cdef float64_t[::1] bit_y

        if not self.initialized:
            raise ValueError("init_node must be called before compute_for_order")
        if order.shape[0] != n:
            raise ValueError("order must have length n_samples")

        bit_w_np = np.zeros(n + 1, dtype=np.float64)
        bit_y_np = np.zeros(n + 1, dtype=np.float64)
        bit_w = bit_w_np
        bit_y = bit_y_np

        for k in range(self.ntaus):
            memset(&bit_w[0], 0, (n + 1) * sizeof(float64_t))
            memset(&bit_y[0], 0, (n + 1) * sizeof(float64_t))
            total_w = 0.0
            total_y = 0.0
            self.left_loss[k, 0] = 0.0
            for pos in range(1, n + 1):
                sample = order[pos - 1]
                rank = self.rank_by_sample[sample]
                total_w += self.weights_by_rank[rank]
                total_y += self.weights_by_rank[rank] * self.values_by_rank[rank]
                self._fenwick_add(bit_w, rank, self.weights_by_rank[rank])
                self._fenwick_add(bit_y, rank,
                                  self.weights_by_rank[rank] * self.values_by_rank[rank])
                loss = self._current_loss(bit_w, bit_y, total_w, total_y, self.tau[k])
                self.left_loss[k, pos] = loss

            memset(&bit_w[0], 0, (n + 1) * sizeof(float64_t))
            memset(&bit_y[0], 0, (n + 1) * sizeof(float64_t))
            total_w = 0.0
            total_y = 0.0
            self.right_loss[k, n] = 0.0
            for pos in range(n - 1, -1, -1):
                sample = order[pos]
                rank = self.rank_by_sample[sample]
                total_w += self.weights_by_rank[rank]
                total_y += self.weights_by_rank[rank] * self.values_by_rank[rank]
                self._fenwick_add(bit_w, rank, self.weights_by_rank[rank])
                self._fenwick_add(bit_y, rank,
                                  self.weights_by_rank[rank] * self.values_by_rank[rank])
                loss = self._current_loss(bit_w, bit_y, total_w, total_y, self.tau[k])
                self.right_loss[k, pos] = loss
        return 0

    cdef inline void _fenwick_add(self, float64_t[::1] bit,
                                  intp_t rank, float64_t value) noexcept nogil:
        cdef intp_t i = rank + 1
        cdef intp_t n = self.n_samples
        while i <= n:
            bit[i] += value
            i += i & -i

    cdef inline float64_t _fenwick_prefix_sum(self, float64_t[::1] bit,
                                              intp_t rank) noexcept nogil:
        cdef intp_t i = rank + 1
        cdef float64_t out = 0.0
        while i > 0:
            out += bit[i]
            i -= i & -i
        return out

    cdef inline intp_t _fenwick_lower_bound(self, float64_t[::1] bit,
                                            float64_t target) noexcept nogil:
        cdef intp_t n = self.n_samples
        cdef intp_t idx = 0
        cdef intp_t bitmask = 1
        cdef intp_t nxt

        while (bitmask << 1) <= n:
            bitmask <<= 1
        while bitmask != 0:
            nxt = idx + bitmask
            if nxt <= n and bit[nxt] < target:
                idx = nxt
                target -= bit[nxt]
            bitmask >>= 1
        if idx >= n:
            return n - 1
        return idx

    cdef inline float64_t _current_loss(self,
                                        float64_t[::1] bit_w,
                                        float64_t[::1] bit_y,
                                        float64_t total_w,
                                        float64_t total_y,
                                        float64_t tau) noexcept nogil:
        cdef intp_t q_rank
        cdef float64_t q, w_le, y_le

        if total_w <= 0.0:
            return 0.0

        q_rank = self._fenwick_lower_bound(bit_w, tau * total_w)
        q = self.values_by_rank[q_rank]
        w_le = self._fenwick_prefix_sum(bit_w, q_rank)
        y_le = self._fenwick_prefix_sum(bit_y, q_rank)
        return (1.0 - tau) * (q * w_le - y_le) + tau * ((total_y - y_le) - q * (total_w - w_le))

    cdef float64_t get_left_loss(self, intp_t tau_id, intp_t pos) noexcept nogil:
        return self.left_loss[tau_id, pos]

    cdef float64_t get_right_loss(self, intp_t tau_id, intp_t pos) noexcept nogil:
        return self.right_loss[tau_id, pos]

    cdef float64_t get_split_loss(self, intp_t tau_id, intp_t pos) noexcept nogil:
        return self.left_loss[tau_id, pos] + self.right_loss[tau_id, pos]

    def get_left_losses(self):
        return np.asarray(self.left_loss_arr)

    def get_right_losses(self):
        return np.asarray(self.right_loss_arr)


def test():
    w = WeightedQuantileCalculator(10, np.array([0.1, 0.5, 0.9]))
    return w.samples.get_value_from_index(-1)


def test2(data, q, weights):
    n = data.size
    w = WeightedQuantileCalculator(n, q)
    for i in range(0, n):
        w.push(data[i], weights[i])
    return w.get_quantile()