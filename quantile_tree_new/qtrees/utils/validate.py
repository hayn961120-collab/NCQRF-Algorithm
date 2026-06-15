import numpy as np


def _validate_tau(tau):
    """Validate and return quantile levels as a 1D float64 NumPy array."""
    if tau is None:
        raise ValueError("tau must be provided.")
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim == 0:
        tau = tau.reshape(1)
    if tau.ndim != 1:
        raise ValueError("tau must be a scalar or 1D array-like.")
    if tau.size == 0:
        raise ValueError("tau must contain at least one quantile level.")
    if np.any(~np.isfinite(tau)):
        raise ValueError("tau must contain only finite values.")
    if np.any((tau <= 0.0) | (tau >= 1.0)):
        raise ValueError("all tau values must lie strictly between 0 and 1.")
    # Keep user order. The Cython criterion handles tau in the provided order.
    return np.ascontiguousarray(tau, dtype=np.float64)
