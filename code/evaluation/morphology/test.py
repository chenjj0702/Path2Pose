import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=1000, n_features=3, n_informative=1, noise=100, random_state=9527)

from scipy.stats import pearsonr

p1 = pearsonr(X[:, 0], y)
p2 = pearsonr(X[:, 1], y)
p3 = pearsonr(X[:, 2], y)

a = 1

# plt.figure()
# a = np.arange(len(y))
# plt.plot(a, X[:, 2])
# plt.plot(a, y)
# plt.show()
