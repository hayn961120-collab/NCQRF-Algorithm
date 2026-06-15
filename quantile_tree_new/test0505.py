from typing import Any, Tuple

import numpy as np

import sklearn
from sklearn.tree import DecisionTreeRegressor
from qtrees.forest import QuantileForest
from simul_utils import gen_simul1, gen_simul2

import matplotlib.pyplot as plt

train, valid, test = gen_simul2(1000)
X = train['data']
y = train['label'].flatten()


# Using bootstrapping 
rf1 = QuantileForest(n_estimators=1000,
                     max_depth=3,
                     tau = [0.1, 0.3, 0.5, 0.7, 0.9],
                     bootstrap=True)


# Using subsampling
# rf1 = QuantileForest(n_estimators=1000,
#                      max_depth=3,
#                      tau = [0.1, 0.3, 0.5, 0.7, 0.9],
#                      bootstrap=False,
#                      max_samples=0.3)
rf1.fit(X, y)
preds = rf1.predict(X)

ind = np.argsort(X.flatten())
plt.scatter(X[ind, :], y[ind])
for i in range(0, preds.shape[1]):
    plt.plot(X[ind, :], preds[ind, i], '-', linewidth = 3)

# plt.plot(X[ind, :], pred_y1[ind], 'r', linewidth = 3)
# plt.plot(X[ind, :], pred_rf_y1[ind], 'g', linewidth = 3)


import pandas as pd
pd.DataFrame(np.concatenate([y[:, None], X], axis = 1), columns = ['y', 'x']).to_csv('sim_test_dat.csv', index = False)