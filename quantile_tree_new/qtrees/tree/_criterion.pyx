# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.string cimport memcpy
from libc.string cimport memset
from libc.string cimport memmove
from libc.stdlib cimport free, qsort, realloc
from libc.math cimport fabs, INFINITY

import numpy as np
cimport numpy as cnp
cnp.import_array()

from scipy.special.cython_special cimport xlogy
from ..utils._quantile cimport WeightedQuantileCalculator
from ..utils._utils cimport safe_realloc


ctypedef struct IndexedYRecord:
    float64_t y
    intp_t sample_index

cdef int _compare_indexed_y(const void* a, const void* b) noexcept nogil:
    cdef IndexedYRecord* ra = <IndexedYRecord*> a
    cdef IndexedYRecord* rb = <IndexedYRecord*> b
    if ra.y < rb.y:
        return -1
    elif ra.y > rb.y:
        return 1
    elif ra.sample_index < rb.sample_index:
        return -1
    elif ra.sample_index > rb.sample_index:
        return 1
    return 0

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to be used on this node
        end : intp_t
            The last sample used on this node

        """
        pass

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]

        Parameters
        ----------
        n_missing: intp_t
            Number of missing values for specific feature.
        """
        pass

    cdef int precompute_split_losses(self, intp_t end_non_missing) except -1 nogil:
        """Optional hook used by expensive order-statistic criteria.

        For ordinary criteria this is a no-op. QuantileLoss overrides this to
        precompute prefix/suffix losses for the current order of sample_indices.
        """
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """
        pass

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : intp_t
            New starting index position of the sample_indices in the right child
        """
        pass

    cdef float64_t node_impurity(self) noexcept nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of sample_indices[start:end]. This is the
        primary function of the criterion class. The smaller the impurity the
        better.
        """
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of sample_indices[start:pos] + the impurity
        of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : float64_t pointer
            The memory address where the impurity of the right child should be
            stored
        """
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of sample_indices[start:end] and save the value into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address where the node value should be stored.
        """
        pass

    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        pass

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints

        This method is implemented in ClassificationCriterion and RegressionCriterion.
        """
        pass

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float64_t
            The initial impurity of the parent node before the split

        impurity_left : float64_t
            The impurity of the left child

        impurity_right : float64_t
            The impurity of the right child

        Return
        ------
        float64_t : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        pass

    cdef inline bint _check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
        float64_t value_left,
        float64_t value_right,
    ) noexcept nogil:
        cdef:
            bint check_lower_bound = (
                (value_left >= lower_bound) &
                (value_right >= lower_bound)
            )
            bint check_upper_bound = (
                (value_left <= upper_bound) &
                (value_right <= upper_bound)
            )
            bint check_monotonic_cst = (
                (value_left - value_right) * monotonic_cst <= 0
            )
        return check_lower_bound & check_upper_bound & check_monotonic_cst

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""


cdef inline void _move_sums_regression(
    RegressionCriterion criterion,
    float64_t[::1] sum_1,
    float64_t[::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef:
        intp_t i
        intp_t n_bytes = criterion.n_outputs * sizeof(float64_t)
        bint has_missing = criterion.n_missing != 0

    if has_missing and put_missing_in_1:
        memcpy(&sum_1[0], &criterion.sum_missing[0], n_bytes)
        for i in range(criterion.n_outputs):
            sum_2[i] = criterion.sum_total[i] - criterion.sum_missing[i]
        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        memset(&sum_1[0], 0, n_bytes)
        # Assigning sum_2 = sum_total for all outputs.
        memcpy(&sum_2[0], &criterion.sum_total[0], n_bytes)
        weighted_n_1[0] = 0.0
        weighted_n_2[0] = criterion.weighted_n_node_samples


cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, intp_t n_outputs, intp_t n_samples, float64_t[::1] tau):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.tau = tau
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.sq_sum_total = 0.0

        self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.sum_right = np.zeros(n_outputs, dtype=np.float64)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t y_ik
        cdef float64_t w_y_ik
        cdef float64_t w = 1.0
        self.sq_sum_total = 0.0
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(float64_t))

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = np.zeros(self.n_outputs, dtype=np.float64)

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, p, k
        cdef float64_t y_ik
        cdef float64_t w_y_ik
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.sum_missing[0], 0, self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_missing[k] += w_y_ik

            self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start
        _move_sums_regression(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left
        )
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        self.pos = self.end
        _move_sums_regression(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left
        )
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t pos = self.pos

        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        cdef intp_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        return (
            (self.sum_left[0] / (2 * self.weighted_n_left)) +
            (self.sum_right[0] / (2 * self.weighted_n_right))
        )

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        cdef:
            float64_t value_left = self.sum_left[0] / self.weighted_n_left
            float64_t value_right = self.sum_right[0] / self.weighted_n_right

        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)


cdef class QuantileLoss(RegressionCriterion):
    r"""Multiple-quantile pinball-loss criterion.

    This version keeps the split criterion as quantile-loss reduction, but avoids
    recomputing child losses from scratch at every split position. For each
    feature-specific sample order, ``precompute_split_losses`` computes exact
    prefix/suffix weighted quantile losses using Fenwick trees. Then
    ``children_impurity`` is an O(K) table lookup, where K is the number of
    quantile levels.
    """

    cdef cnp.ndarray left_child
    cdef cnp.ndarray right_child
    cdef void** left_child_ptr
    cdef void** right_child_ptr
    cdef float64_t[::1] node_quantiles

    cdef IndexedYRecord* sort_records
    cdef intp_t* rank_by_sample
    cdef float64_t* values_by_rank
    cdef float64_t* weights_by_rank
    cdef float64_t* bit_w
    cdef float64_t* bit_y
    cdef float64_t* left_loss
    cdef float64_t* right_loss
    cdef float64_t* left_quantiles
    cdef float64_t* right_quantiles
    cdef float64_t* node_losses
    cdef intp_t loss_stride
    cdef bint losses_precomputed

    def __cinit__(self, intp_t n_outputs, intp_t n_samples, float64_t[::1] tau):
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.tau = tau
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.loss_stride = n_samples + 1
        self.losses_precomputed = False

        self.sort_records = NULL
        self.rank_by_sample = NULL
        self.values_by_rank = NULL
        self.weights_by_rank = NULL
        self.bit_w = NULL
        self.bit_y = NULL
        self.left_loss = NULL
        self.right_loss = NULL
        self.left_quantiles = NULL
        self.right_quantiles = NULL
        self.node_losses = NULL

        self.node_quantiles = np.zeros(n_outputs, dtype=np.float64)

        self.sort_records = <IndexedYRecord*> realloc(self.sort_records, n_samples * sizeof(IndexedYRecord))
        if self.sort_records == NULL:
            raise MemoryError()
        safe_realloc(&self.rank_by_sample, n_samples)
        safe_realloc(&self.values_by_rank, n_samples)
        safe_realloc(&self.weights_by_rank, n_samples)
        safe_realloc(&self.bit_w, n_samples + 1)
        safe_realloc(&self.bit_y, n_samples + 1)
        safe_realloc(&self.left_loss, n_outputs * (n_samples + 1))
        safe_realloc(&self.right_loss, n_outputs * (n_samples + 1))
        safe_realloc(&self.left_quantiles, n_outputs * (n_samples + 1))
        safe_realloc(&self.right_quantiles, n_outputs * (n_samples + 1))
        safe_realloc(&self.node_losses, n_outputs)

        # Kept for backward compatibility with old pickles/tests that access
        # these object arrays, but the fast path below does not use them.
        self.left_child = np.empty(1, dtype='object')
        self.right_child = np.empty(1, dtype='object')
        self.left_child[0] = WeightedQuantileCalculator(n_samples, tau)
        self.right_child[0] = WeightedQuantileCalculator(n_samples, tau)
        self.left_child_ptr = <void**> cnp.PyArray_DATA(self.left_child)
        self.right_child_ptr = <void**> cnp.PyArray_DATA(self.right_child)

    def __dealloc__(self):
        free(self.sort_records)
        free(self.rank_by_sample)
        free(self.values_by_rank)
        free(self.weights_by_rank)
        free(self.bit_w)
        free(self.bit_y)
        free(self.left_loss)
        free(self.right_loss)
        free(self.left_quantiles)
        free(self.right_quantiles)
        free(self.node_losses)

    cdef inline void _fenwick_clear(self) noexcept nogil:
        memset(self.bit_w, 0, (self.n_node_samples + 1) * sizeof(float64_t))
        memset(self.bit_y, 0, (self.n_node_samples + 1) * sizeof(float64_t))

    cdef inline void _fenwick_add(self, float64_t* bit, intp_t rank, float64_t value) noexcept nogil:
        cdef intp_t i = rank + 1
        cdef intp_t n = self.n_node_samples
        while i <= n:
            bit[i] += value
            i += i & -i

    cdef inline float64_t _fenwick_prefix_sum(self, float64_t* bit, intp_t rank) noexcept nogil:
        cdef intp_t i = rank + 1
        cdef float64_t out = 0.0
        while i > 0:
            out += bit[i]
            i -= i & -i
        return out

    cdef inline intp_t _fenwick_lower_bound(self, float64_t* bit, float64_t target) noexcept nogil:
        cdef intp_t n = self.n_node_samples
        cdef intp_t idx = 0
        cdef intp_t bitmask = 1
        cdef intp_t nxt

        if target <= 0.0:
            return 0

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

    cdef inline float64_t _loss_from_fenwick(
        self,
        float64_t total_w,
        float64_t total_y,
        float64_t tau,
        float64_t* q_out,
    ) noexcept nogil:
        cdef intp_t q_rank
        cdef float64_t q, w_le, y_le

        if total_w <= 0.0:
            q_out[0] = 0.0
            return 0.0

        q_rank = self._fenwick_lower_bound(self.bit_w, tau * total_w)
        q = self.values_by_rank[q_rank]
        q_out[0] = q
        w_le = self._fenwick_prefix_sum(self.bit_w, q_rank)
        y_le = self._fenwick_prefix_sum(self.bit_y, q_rank)

        return ((1.0 - tau) * (q * w_le - y_le)
                + tau * ((total_y - y_le) - q * (total_w - w_le)))

    cdef inline float64_t _node_loss_from_sorted(self, float64_t tau, float64_t* q_out) noexcept nogil:
        cdef intp_t r, q_rank
        cdef float64_t target = tau * self.weighted_n_node_samples
        cdef float64_t cum_w = 0.0
        cdef float64_t q = 0.0
        cdef float64_t loss = 0.0
        cdef float64_t y, w

        if self.weighted_n_node_samples <= 0.0:
            q_out[0] = 0.0
            return 0.0

        q_rank = self.n_node_samples - 1
        for r in range(self.n_node_samples):
            cum_w += self.weights_by_rank[r]
            if cum_w >= target:
                q_rank = r
                break

        q = self.values_by_rank[q_rank]
        q_out[0] = q

        for r in range(self.n_node_samples):
            y = self.values_by_rank[r]
            w = self.weights_by_rank[r]
            if y < q:
                loss += w * (1.0 - tau) * (q - y)
            else:
                loss += w * tau * (y - q)
        return loss

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        cdef intp_t i, p, r, k
        cdef float64_t w = 1.0
        cdef float64_t q

        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.losses_precomputed = False

        for p in range(start, end):
            i = sample_indices[p]
            if sample_weight is not None:
                w = sample_weight[i]
            else:
                w = 1.0
            self.sort_records[p - start].y = y[i, 0]
            self.sort_records[p - start].sample_index = i
            self.weighted_n_node_samples += w

        qsort(self.sort_records, self.n_node_samples, sizeof(IndexedYRecord), _compare_indexed_y)

        for r in range(self.n_node_samples):
            i = self.sort_records[r].sample_index
            self.rank_by_sample[i] = r
            self.values_by_rank[r] = self.sort_records[r].y
            if sample_weight is not None:
                self.weights_by_rank[r] = sample_weight[i]
            else:
                self.weights_by_rank[r] = 1.0

        for k in range(self.n_outputs):
            self.node_losses[k] = self._node_loss_from_sorted(self.tau[k], &q)
            self.node_quantiles[k] = q

        self.reset()
        return 0

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        if n_missing == 0:
            return
        with gil:
            raise ValueError("missing values are not supported for QuantileLoss.")

    cdef int reset(self) except -1 nogil:
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        self.pos = self.end
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t i, p
        cdef float64_t w

        if new_pos < self.pos:
            self.reset()

        for p in range(self.pos, new_pos):
            i = sample_indices[p]
            if sample_weight is not None:
                w = sample_weight[i]
            else:
                w = 1.0
            self.weighted_n_left += w

        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = new_pos
        return 0

    cdef int precompute_split_losses(self, intp_t end_non_missing) except -1 nogil:
        """Precompute exact child quantile losses for current sample order.

        For QuantileLoss, missing values are unsupported, so end_non_missing must
        equal self.end in normal usage. The argument is kept to match splitter
        calls and to make accidental missing-value use fail explicitly through
        init_missing().
        """
        cdef intp_t n = end_non_missing - self.start
        cdef intp_t k, offset, sample, rank
        cdef float64_t total_w, total_y, w, yv, loss, q
        cdef intp_t base

        if n != self.n_node_samples:
            with gil:
                raise ValueError("QuantileLoss does not support missing values in fast precomputation.")

        for k in range(self.n_outputs):
            base = k * self.loss_stride

            self._fenwick_clear()
            total_w = 0.0
            total_y = 0.0
            self.left_loss[base] = 0.0
            self.left_quantiles[base] = 0.0

            for offset in range(1, n + 1):
                sample = self.sample_indices[self.start + offset - 1]
                rank = self.rank_by_sample[sample]
                w = self.weights_by_rank[rank]
                yv = self.values_by_rank[rank]
                total_w += w
                total_y += w * yv
                self._fenwick_add(self.bit_w, rank, w)
                self._fenwick_add(self.bit_y, rank, w * yv)
                loss = self._loss_from_fenwick(total_w, total_y, self.tau[k], &q)
                self.left_loss[base + offset] = loss
                self.left_quantiles[base + offset] = q

            self._fenwick_clear()
            total_w = 0.0
            total_y = 0.0
            self.right_loss[base + n] = 0.0
            self.right_quantiles[base + n] = 0.0

            for offset in range(n - 1, -1, -1):
                sample = self.sample_indices[self.start + offset]
                rank = self.rank_by_sample[sample]
                w = self.weights_by_rank[rank]
                yv = self.values_by_rank[rank]
                total_w += w
                total_y += w * yv
                self._fenwick_add(self.bit_w, rank, w)
                self._fenwick_add(self.bit_y, rank, w * yv)
                loss = self._loss_from_fenwick(total_w, total_y, self.tau[k], &q)
                self.right_loss[base + offset] = loss
                self.right_quantiles[base + offset] = q

        self.losses_precomputed = True
        return 0

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        cdef intp_t k
        for k in range(self.n_outputs):
            dest[k] = <float64_t> self.node_quantiles[k]

    cdef inline float64_t middle_value(self) noexcept nogil:
        cdef intp_t offset = self.pos - self.start
        if self.losses_precomputed:
            return (self.left_quantiles[offset] + self.right_quantiles[offset]) / 2.0
        return self.node_quantiles[0]

    cdef inline bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        cdef intp_t offset = self.pos - self.start
        cdef float64_t value_left = self.left_quantiles[offset]
        cdef float64_t value_right = self.right_quantiles[offset]
        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)

    cdef float64_t node_impurity(self) noexcept nogil:
        cdef intp_t k
        cdef float64_t impurity = 0.0
        for k in range(self.n_outputs):
            impurity += self.node_losses[k]
        return impurity / (self.weighted_n_node_samples * self.n_outputs)

    cdef void children_impurity(self, float64_t* p_impurity_left,
                                float64_t* p_impurity_right) noexcept nogil:
        cdef intp_t k
        cdef intp_t offset = self.pos - self.start
        cdef intp_t base
        cdef float64_t impurity_left = 0.0
        cdef float64_t impurity_right = 0.0

        if self.weighted_n_left <= 0.0:
            p_impurity_left[0] = INFINITY
        else:
            for k in range(self.n_outputs):
                base = k * self.loss_stride
                impurity_left += self.left_loss[base + offset]
            p_impurity_left[0] = impurity_left / (self.weighted_n_left * self.n_outputs)

        if self.weighted_n_right <= 0.0:
            p_impurity_right[0] = INFINITY
        else:
            for k in range(self.n_outputs):
                base = k * self.loss_stride
                impurity_right += self.right_loss[base + offset]
            p_impurity_right[0] = impurity_right / (self.weighted_n_right * self.n_outputs)

def test_criterion():
    loss = QuantileLoss(4, 100, np.array([0.1, 0.5, 0.9]))
    y = np.random.randn(100)[:, None]
    weights = np.ones(100, dtype = np.float64)
    ind = np.ones(100, dtype = np.intp)
    loss.init(y, weights, 100, ind, 0, 100)
    return loss.node_quantiles