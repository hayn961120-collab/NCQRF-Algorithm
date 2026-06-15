# qtrees install/test instructions

This directory is arranged as an installable Python package.

## Recommended install

Use the existing environment where NumPy, SciPy, scikit-learn and Cython are already installed:

```bash
cd quantile_tree_installable
python -m pip uninstall -y qtrees
rm -rf build dist *.egg-info
find . -name "*.so" -delete
find . -name "*.c" -delete
find . -name "*.cpp" -delete
python -m pip install -v -e . --no-build-isolation
```

`--no-build-isolation` is recommended because the Cython sources cimport SciPy and NumPy headers. It avoids pip creating a temporary build environment that may not contain matching versions.

## Smoke test

```bash
python tests/test_smoke.py
```

or directly:

```python
import numpy as np
from qtrees.tree import QuantileTree

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4)).astype(np.float32)
y = X[:, 0] + 0.25 * rng.normal(size=200)

est = QuantileTree(max_depth=3, min_samples_leaf=5, tau=[0.1, 0.5, 0.9], random_state=0)
est.fit(X, y)
print(est.predict(X[:5]))
```

## What was added

Missing package/build files were added:

- `pyproject.toml`
- `setup.py`
- root package `qtrees/__init__.py`
- `qtrees/utils/__init__.py`
- `qtrees/utils/validate.py`
- optional Meson build files: root `meson.build`, `qtrees/meson.build`, `qtrees/utils/meson.build`, `qtrees/tree/meson.build`

The default install path uses setuptools/Cython through `pyproject.toml` and `setup.py`.
