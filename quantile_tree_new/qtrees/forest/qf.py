"""
Quantile random forest.

Those methods include quantile random forest and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremely randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

Single and multi-output problems are both handled.
"""

# Authors: Beomjin Park
import threading
import numpy as np
from warnings import catch_warnings, simplefilter, warn

from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse

from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)
from sklearn.exceptions import DataConversionWarning

from ..tree.tree import QuantileTree
__all__ = ['BaseQuantileForest', "QuantileForest"]

# =============================================================================
# Types and constants
# =============================================================================

from numbers import Integral, Real
from typing import Tuple, TypeVar
from numpy import float32 as DTYPE

T = TypeVar("T", bound = np.generic, covariant = True)

Vector = np.ndarray[Tuple[int], np.dtype[T]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T]]

MAX_INT = np.iinfo(np.int32).max
    

def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0.0, 1.0]`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, Integral):
        if max_samples <= 0:
            raise ValueError("`max_samples` must be greater than 0.")
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return int(max_samples)

    if isinstance(max_samples, Real):
        if max_samples <= 0.0 or max_samples > 1.0:
            raise ValueError(
                "`max_samples` as a float must be in the range (0.0, 1.0]."
            )
        return max(round(n_samples * max_samples), 1)
    
    raise TypeError("`max_samples` must be int, float, or None.")


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(
        0, n_samples, n_samples_bootstrap, dtype=np.int32
    )

    return sample_indices


def _generate_sample_indices_without_replacement(
    random_state,
    n_samples,
    n_samples_subsample,
    ):
    """
    Generate sample indices without replacement for subsampling.
    """
    random_instance = check_random_state(random_state)
    if n_samples_subsample > n_samples:
        raise ValueError(
            "n_samples_subsample cannot be larger than n_samples when sampling without replacement."
        )
    sample_indices = random_instance.choice(
        n_samples,
        size=n_samples_subsample,
        replace=False,
    ).astype(np.int32, copy=False)
    return sample_indices    


def _generate_unsampled_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to forest._set_oob_score function."""
    sample_indices = _generate_sample_indices(
        random_state, n_samples, n_samples_bootstrap
    )
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]

    return unsampled_indices


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    missing_values_in_feature_mask=None,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    n_samples = X.shape[0]
    if bootstrap:
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree._fit(
            X,
            y,
            sample_weight=curr_sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )
    elif n_samples_bootstrap is not None:
        if n_samples_bootstrap > n_samples:
            raise ValueError(
                'n_samples_bootstrap cannot be larger than n_samples when bootstrap=False'
            )
            
        curr_sample_weight = np.zeros((n_samples,), dtype=np.float64)
        indices = _generate_sample_indices_without_replacement(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        if sample_weight is None:
            curr_sample_weight[indices] = 1.0
        else:
            curr_sample_weight[indices] = sample_weight[indices]
        
        tree._fit(
            X,
            y,
            sample_weight=curr_sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )
    else:
        tree._fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=False,
            missing_values_in_feature_mask=missing_values_in_feature_mask,
        )

    return tree

def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]
                
                
class BaseQuantileForest(BaseEnsemble):
    """
    Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super(BaseQuantileForest, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        
        
    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(delayed(tree.apply)(X, check_input=False) for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        .. versionadded:: 0.18

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.

        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        X = self._validate_X_predict(X)
        indicators = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(
            delayed(tree.decision_path)(X, check_input=False)
            for tree in self.estimators_
        )

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def _fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) 
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            ensure_all_finite=False,
        )
        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.
        estimator = type(self.estimator)()
        missing_values_in_feature_mask = (
            estimator._compute_missing_values_in_feature_mask(
                X, estimator_name=self.__class__.__name__
            )
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self._n_samples = y.shape[0]

        # if not self.bootstrap and self.max_samples is not None:
        #     raise ValueError(
        #         "`max_sample` cannot be set if `bootstrap=False`. "
        #         "Either switch to `bootstrap=True` or set "
        #         "`max_sample=None`."
        #     )
        # if self.bootstrap:
        if self.max_samples is None and not self.bootstrap:
            n_samples_bootstrap = None
        else:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        # else:
            # n_samples_bootstrap = self.max_samples

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score and (
            n_more_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        return self

    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        scoring_function : callable, default=None
            Scoring function for OOB score. Default depends on whether
            this is a regression (R2 score) or classification problem
            (accuracy score).
        """

    def _compute_oob_predictions(self, X, y):
        """Compute and set the OOB score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_classes, n_outputs) or \
                (n_samples, 1, n_outputs)
            The OOB predictions.
        """
        
        check_is_fitted(self)
        
        # Prediction requires X to be in CSR format
        if issparse(X):
            X = X.tocsr()

        n_samples = y.shape[0]
        n_taus = len(self.estimators_[0].tau)
        
        # for regression, n_classes_ does not exist and we create an empty
        # axis to be consistent with the classification case and make
        # the array operations compatible with the 2 settings
        oob_pred_shape = (n_samples, 1, n_taus)

        oob_pred = np.zeros(shape=oob_pred_shape, dtype=np.float64)
        n_oob_pred = np.zeros((n_samples, n_taus), dtype=np.int64)

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples,
            self.max_samples,
        )
        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state,
                n_samples,
                n_samples_bootstrap,
            )

            y_pred = self._get_oob_predictions(estimator, X[unsampled_indices, :])
            oob_pred[unsampled_indices, ...] += y_pred
            n_oob_pred[unsampled_indices, :] += 1

        for k in range(n_taus):
            if (n_oob_pred == 0).any():
                warn(
                    (
                        "Some inputs do not have OOB scores. This probably means "
                        "too few trees were used to compute any reliable OOB "
                        "estimates."
                    ),
                    UserWarning,
                )
                n_oob_pred[n_oob_pred == 0] = 1
            oob_pred[..., k] /= n_oob_pred[..., [k]]

        return oob_pred

    def _validate_y_class_weight(self, y):
        # Default implementation
        return y, None

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)
        if self.estimators_[0]._support_missing_values(X):
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
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based sparse matrices")
        return X

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(getattr)(tree, "feature_importances_")
            for tree in self.estimators_
            if tree.tree_.node_count > 1
        )

        if not all_importances:
            return np.zeros(self.n_features_in_, dtype=np.float64)

        all_importances = np.mean(all_importances, axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        for tree in self.estimators_:
            # tree.random_state is actually an immutable integer seed rather
            # than a mutable RandomState instance, so it's safe to use it
            # repeatedly when calling this property.
            seed = tree.random_state
            
            if self.bootstrap:
                # Operations accessing random_state must be performed identically
                # to those in `_parallel_build_trees()`
                yield _generate_sample_indices(
                    seed, self._n_samples, self._n_samples_bootstrap
                )
            elif self.max_samples is not None:
                yield _generate_sample_indices_without_replacement(
                    seed, self._n_samples, self._n_samples_bootstrap,
                )
            else:
                yield np.arange(self._n_samples, dtype=np.int32)
            
                
    @property
    def estimators_samples_(self):
        """The subset of drawn samples for each base estimator.

        Returns a dynamically generated list of indices identifying
        the samples used for fitting each member of the ensemble, i.e.,
        the in-bag samples.

        Note: the list is re-created at each call to the property in order
        to reduce the object memory footprint by not storing the sampling
        data. Thus fetching the property may be slower than expected.
        """
        return [sample_indices for sample_indices in self._get_estimators_indices()]


class QuantileForest(BaseQuantileForest):
    """
    Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(
        self,
        n_estimators = 100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        tau=None,
    ):
        super(QuantileForest, self).__init__(
            estimator = QuantileTree(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
                "tau",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.tau = tau

    def fit(self, X, y, sample_weight=None):
        
        if y.ndim == 2 and y.shape[1] > 1:
            warn(
                (
                    "Multiple outputs are not supported in QuantileForest."
                ),
                DataConversionWarning,
            )
            
        super(QuantileForest, self)._fit(
            X = X,
            y = y,
            sample_weight = sample_weight
        )
        return self
    
    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        n_taus = len(self.tau)
        if n_taus > 1:
            y_hat = np.zeros((X.shape[0], n_taus), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # single output regression
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # multioutput regression
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred

    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        scoring_function : callable, default=None
            Scoring function for OOB score. Defaults to `r2_score`.
        """
        self.oob_prediction_ = super()._compute_oob_predictions(X, y).squeeze(axis=1)
        if self.oob_prediction_.shape[-1] == 1:
            # drop the n_outputs axis if there is a single output
            self.oob_prediction_ = self.oob_prediction_.squeeze(axis=-1)

        # if scoring_function is None:
        #     scoring_function = r2_score

        # self.oob_score_ = scoring_function(y, self.oob_prediction_)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features), dtype=DTYPE
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features), dtype=np.intp
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        target_features = np.asarray(target_features, dtype=np.intp, order="C")
        averaged_predictions = np.zeros(
            shape=grid.shape[0], dtype=np.float64, order="C"
        )

        for tree in self.estimators_:
            # Note: we don't sum in parallel because the GIL isn't released in
            # the fast method.
            tree.tree_.compute_partial_dependence(
                grid, target_features, averaged_predictions
            )
        # Average over the forest
        averaged_predictions /= len(self.estimators_)

        return averaged_predictions

