"""
This module gathers a quantile tree method.
"""

# Authors: Beomjin Park

import numpy as np
import numbers
from math import ceil

from scipy.sparse import issparse

from sklearn.base import clone, BaseEstimator
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import (
    _assert_all_finite_element_wise,
     _check_n_features,
    _check_sample_weight,
    assert_all_finite,
    check_is_fitted,
    validate_data,
)

from . import _criterion, _splitter, _tree
from ._criterion import QuantileLoss

from ._splitter import (
    BestSplitter,
    RandomSplitter,
    BestSparseSplitter,
    RandomSparseSplitter,
)

from ._tree import (
    BestFirstTreeBuilder,
    DepthFirstTreeBuilder,
    Tree,
    _build_pruned_tree_ccp,
    ccp_pruning_path,
)

from ..utils._utils import _any_isnan_axis0
from ..utils.validate import _validate_tau


__all__ = ["BaseQuantileTree", "QuantileTree"]

# =============================================================================
# Types and constants
# =============================================================================

from numbers import Integral, Real
from typing import Tuple, TypeVar
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE
from numpy.typing import ArrayLike
from numpy.random import RandomState

T = TypeVar("T", bound = np.generic, covariant = True)

Vector = np.ndarray[Tuple[int], np.dtype[T]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T]]

DENSE_SPLITTERS = {
    "best": BestSplitter,
    "random": RandomSplitter
}
SPARSE_SPLITTERS = {
    "best": BestSparseSplitter,
    "random": RandomSparseSplitter
}

# =============================================================================
# Base quantile tree
# =============================================================================


class BaseQuantileTree(BaseEstimator):
    
    def __init__(
        self,
        splitter: str = 'best',
        max_depth: Integral | None = None,
        min_samples_split: Integral | Real = 2,
        min_samples_leaf: Integral | Real = 1,
        min_weight_fraction_leaf: Real = 0.0,
        max_features: str | Integral | Real | None = None,
        max_leaf_nodes: Integral | None = None,
        random_state: Integral | RandomState | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: Real = 0.0,
        monotonic_cst: ArrayLike | None = None,
        tau: ArrayLike = None
    ) -> None:
        # Add range validation code later
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.tau = tau
    
    def get_depth(self):
        """Return the depth of the decision tree.

        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    def _support_missing_values(self, X):
        return (
            not issparse(X)
            and self.monotonic_cst is None
        )

    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        """Return boolean mask denoting if there are missing values for each feature.

        This method also ensures that X is finite.

        Parameter
        ---------
        X : array-like of shape (n_samples, n_features), dtype=DOUBLE
            Input data.

        estimator_name : str or None, default=None
            Name to use when raising an error. Defaults to the class name.

        Returns
        -------
        missing_values_in_feature_mask : ndarray of shape (n_features,), or None
            Missing value mask. If missing values are not supported or there
            are no missing values, return None.
        """
        estimator_name = estimator_name or self.__class__.__name__
        common_kwargs = dict(estimator_name=estimator_name, input_name="X")

        if not self._support_missing_values(X):
            assert_all_finite(X, **common_kwargs)
            return None

        with np.errstate(over="ignore"):
            overall_sum = np.sum(X)

        if not np.isfinite(overall_sum):
            # Raise a ValueError in case of the presence of an infinite element.
            _assert_all_finite_element_wise(X, xp=np, allow_nan=True, **common_kwargs)

        # If the sum is not nan, then there are no missing values
        if not np.isnan(overall_sum):
            return None

        missing_values_in_feature_mask = _any_isnan_axis0(X)
        return missing_values_in_feature_mask
        
    def _fit(
        self,
        X,
        y,
        sample_weight: ArrayLike | None = None,
        check_input: bool = True,
        missing_values_in_feature_mask: Vector | None = None,
    ):
        random_state = check_random_state(self.random_state)

        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.

            # _compute_missing_values_in_feature_mask will check for finite values and
            # compute the missing mask if the tree supports missing values
            check_X_params = dict(
                dtype=DTYPE, accept_sparse="csc", ensure_all_finite=False
            )
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = validate_data(
                self, X, y, validate_separately=(check_X_params, check_y_params)
            )

            missing_values_in_feature_mask = (
                self._compute_missing_values_in_feature_mask(X)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based sparse matrices"
                    )

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape
        
        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=DOUBLE)

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree
        tau = _validate_tau(self.tau)
        self.n_taus = len(tau)
         
        
        criterion = QuantileLoss(self.n_taus, n_samples, tau)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if self.monotonic_cst is None:
            monotonic_cst = None
        else:
            if self.n_taus > 1:
                raise ValueError(
                    "Monotonicity constraints are not supported with multiple quantile levels"
                )
            # Check to correct monotonicity constraint' specification,
            # by applying element-wise logical conjunction
            # Note: we do not cast `np.asarray(self.monotonic_cst, dtype=np.int8)`
            # straight away here so as to generate error messages for invalid
            # values using the original values prior to any dtype related conversion.
            monotonic_cst = np.asarray(self.monotonic_cst)
            if monotonic_cst.shape[0] != X.shape[1]:
                raise ValueError(
                    "monotonic_cst has shape {} but the input data "
                    "X has {} features.".format(monotonic_cst.shape[0], X.shape[1])
                )
            valid_constraints = np.isin(monotonic_cst, (-1, 0, 1))
            if not np.all(valid_constraints):
                unique_constaints_value = np.unique(monotonic_cst)
                raise ValueError(
                    "monotonic_cst must be None or an array-like of -1, 0 or 1, but"
                    f" got {unique_constaints_value}"
                )
            monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)

        splitter = SPLITTERS[self.splitter](
            criterion,
            self.max_features_,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
            monotonic_cst,
        )

        self.tree_ = Tree(
            self.n_features_in_,
            # TODO: tree shouldn't need this in this case
            np.array([1] * self.n_taus, dtype=np.intp),
            self.n_taus,
        )

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)
        self._prune_tree()

        return self
    
    def _validate_X_predict(self, X, check_input):
        """Validate the training data on predict (probabilities)."""
        if check_input:
            if self._support_missing_values(X):
                ensure_all_finite = "allow-nan"
            else:
                ensure_all_finite = True
            X = validate_data(
                self,
                X,
                dtype=DTYPE,
                accept_sparse="csr",
                reset=False,
                ensure_all_finite=ensure_all_finite,
            )
            if issparse(X) and (
                X.indices.dtype != np.intc or X.indptr.dtype != np.intc
            ):
                raise ValueError("No support for np.int64 index based sparse matrices")
        else:
            # The number of features is checked regardless of `check_input`
            _check_n_features(self, X, reset=False)
        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred_y = self.tree_.predict(X)
        
        if self.n_taus == 1:
            return pred_y[:, 0]

        else:
            return pred_y[:, :, 0]
        
        
    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    def _prune_tree(self):
        """Prune tree using Minimal Cost-Complexity Pruning."""
        check_is_fitted(self)

        if self.ccp_alpha == 0.0:
            return

        # build pruned tree
        pruned_tree = Tree(
            self.n_features_in_,
            # TODO: the tree shouldn't need this param
            np.array([1] * self.n_taus, dtype=np.intp),
            self.n_taus,
        )
        _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

        self.tree_ = pruned_tree

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """Compute the pruning path during Minimal Cost-Complexity Pruning.

        See :ref:`minimal_cost_complexity_pruning` for details on the pruning
        process.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        Returns
        -------
        ccp_path : :class:`~sklearn.utils.Bunch`
            Dictionary-like object, with the following attributes.

            ccp_alphas : ndarray
                Effective alphas of subtree during pruning.

            impurities : ndarray
                Sum of the impurities of the subtree leaves for the
                corresponding alpha value in ``ccp_alphas``.
        """
        est = clone(self).set_params(ccp_alpha=0.0)
        est.fit(X, y, sample_weight=sample_weight)
        return Bunch(**ccp_pruning_path(est.tree_))

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).
        """
        check_is_fitted(self)

        return self.tree_.compute_feature_importances()


class QuantileTree(BaseQuantileTree):
    
    def __init__(
        self,
        splitter: str = 'best',
        max_depth: Integral | None = None,
        min_samples_split: Integral | Real = 2,
        min_samples_leaf: Integral | Real = 1,
        min_weight_fraction_leaf: Real = 0.0,
        max_features: str | Integral | Real | None = None,
        max_leaf_nodes: Integral | None = None,
        random_state: Integral | RandomState | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: Real = 0.0,
        monotonic_cst: ArrayLike | None = None,
        tau: ArrayLike = None
    ) -> None:
        super(QuantileTree, self).__init__(
            splitter = splitter,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features,
            max_leaf_nodes = max_leaf_nodes,
            random_state = random_state,
            min_impurity_decrease = min_impurity_decrease,
            ccp_alpha = ccp_alpha,
            monotonic_cst = monotonic_cst,
            tau = tau,
        )
        
    def fit(
        self,
        X,
        y,
        sample_weight: ArrayLike | None = None,
        check_input: bool = True
        ):
        
        super(QuantileTree, self)._fit(
            X,
            y,
            sample_weight = sample_weight,
            check_input = check_input
        )
        
        return self