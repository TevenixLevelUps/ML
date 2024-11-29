import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification


X, y = make_classification(n_samples=1000,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_classes=2,
                           class_sep=2,
                           random_state=1)


plt.scatter(X[:, 0][y==0], X[:, 1][y==0], marker='o', c='r', s=100)
plt.scatter(X[:, 0][y==1], X[:, 1][y==1], marker='x', c='b', s=100)
plt.show()
