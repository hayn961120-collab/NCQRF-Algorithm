import numpy as np
from qtrees.tree import QuantileTree
import datetime

rng = np.random.default_rng(0)
n = 2000
X = rng.normal(size=(n, 1)).astype(np.float32)
y = X[:, 0] + 0.25 * rng.normal(size=n)

s = datetime.datetime.now()
est = QuantileTree(max_depth=5, min_samples_leaf=5, tau=[0.1, 0.3, 0.5, 0.7, 0.9], random_state=0)
est.fit(X, y)
pred = est.predict(X)
e = datetime.datetime.now()
print(e - s)


import matplotlib.pyplot as plt
plt.scatter(X, y)
for i in range(pred.shape[1]):
    plt.scatter(X, pred[:, i])


