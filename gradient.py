import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

adver_data = pd.read_csv('data/housing.csv')
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

data = pd.DataFrame(adver_data, columns=features)
X_filter = data[data["median_house_value"] != 500001.0]  # delete anomal
mean = data.mean()  # expectation
std = data.std()  # dispersion
X_filter = X_filter[(X_filter < mean + 3 * std) & (X_filter > mean - 3 * std)]  # delete outliers
y = X_filter["median_house_value"].to_numpy().reshape(-1, 1)
X = X_filter.drop("median_house_value", axis=1)
X.fillna(X.mean(), inplace=True)
X = X.values
means, stds = np.mean(X, axis=0), np.std(X, axis=0)

for num in range(len(X)):
    for i in range(0, 8):
        X[num][i] = (X[num][i] - means[i]) / stds[i]

# adding column of ones
b = np.ones(X.shape[0])
d = b.reshape((X.shape[0], 1))
X = np.hstack((d, X))


class Model(object):
    def __init__(self):
        self.w = np.random.randn(9, 1)

    # func of MSE
    def error(self, y, y_pred):
        s, s_pred = pd.Series(y), pd.Series(y_pred)
        return float(sum((s - s_pred) ** 2) / (2 * len(y)))

    # predict func
    def linear_pred(self, X):
        return np.dot(X, self.w)

    # stochastic gradient step
    def stochastic_gradient_step(self, X, y, train_ind, eta=500):
        grad = X[train_ind].reshape(-1, 1).dot(X[train_ind].T.dot(self.w).reshape(-1, 1) - y[train_ind].reshape(-1, 1))
        return self.w - ((eta / len(y)) * np.array(grad))

    # batch gradient step
    def batch_gradient_step(self, X, y, indices, eta=500):
        X_batch = np.array(X)[indices]
        y_batch = np.array(y)[indices]
        grad = X_batch.T.dot(X_batch.dot(self.w) - y_batch)
        return self.w - ((eta / len(y)) * np.array(grad))

    # full gradient step
    def full_gradient_step(self, X, y, eta=500):
        grad = X.T.dot(X.dot(self.w) - y)
        return self.w - ((eta / len(y)) * np.array(grad))

    # cycle of stochastic gradient
    def stochastic_gradient_descent(self, X, y, w_init, eta=100, max_iter=1e4, min_weight_dist=1e-8):
        weight_dist = np.inf
        self.w = np.array(w_init).reshape(9, 1)
        errors = []
        iter_num = 0

        # main cycle
        while weight_dist > min_weight_dist and iter_num < max_iter:
            # choose a random index
            random_ind = np.random.randint(X.shape[0])
            new_w = self.stochastic_gradient_step(X, y, random_ind, eta)
            iter_num += 1
            fr_cst = self.linear_pred(X)
            weight_dist = np.linalg.norm(new_w - self.w)
            errors.append(self.error(y[:, 0], fr_cst.flatten()))
            self.w = new_w

        return errors, eta

    # cycle of full gradient
    def full_gradient_descent(self, X, y, w_init, eta=0.05, max_iter=1e4, min_weight_dist=1e-8):
        weight_dist = np.inf
        self.w = np.array(w_init).reshape(9, 1)
        errors = []
        iter_num = 0

        # main cycle
        while weight_dist > min_weight_dist and iter_num < max_iter:
            new_w = self.full_gradient_step(X, y, eta)
            iter_num += 1
            fr_cst = self.linear_pred(X)
            weight_dist = np.linalg.norm(new_w - self.w)
            errors.append(self.error(y[:, 0], fr_cst.flatten()))
            self.w = new_w

        return errors

    # cycle of batch gradient
    def batch_gradient_descent(self, X, y, w_init, eta=100, k=20, max_iter=1e4, min_weight_dist=1e-8):
        weight_dist = np.inf
        self.w = np.array(w_init).reshape(9, 1)
        errors = []
        iter_num = 0

        # main cycle
        while weight_dist > min_weight_dist and iter_num < max_iter:
            # choose a random k indexes
            indices = np.random.choice(len(X), k, replace=False)
            new_w = self.batch_gradient_step(X, y, indices, eta)
            iter_num += 1
            fr_cst = self.linear_pred(X)
            weight_dist = np.linalg.norm(new_w - self.w)
            errors.append(self.error(y[:, 0], fr_cst.flatten()))
            self.w = new_w

        return errors


m_stoch = Model()
m_batch = Model()
m_full = Model()

stoch_errors_by_iter, eta = m_stoch.stochastic_gradient_descent(X=X, y=y, w_init=(np.ones(9) * 0), max_iter=1000)
batch_errors_by_iter = m_batch.batch_gradient_descent(X=X, y=y, w_init=(np.ones(9) * 0), k=20, max_iter=1000)
full_errors_by_iter = m_full.full_gradient_descent(X=X, y=y, w_init=(np.ones(9) * 0), max_iter=1000)

# prediction graph
y_pred_stoch = m_stoch.linear_pred(X)
plt.figure(figsize=[12, 8])
plt.scatter(y_pred_stoch, y, c='gray', alpha=0.3)
plt.plot(y, y, 'b')
plt.title(f"stochastic_gradient_descent {eta}")
plt.show()

y_pred_batch = m_batch.linear_pred(X)
plt.figure(figsize=[12, 8])
plt.scatter(y_pred_batch, y, c='gray', alpha=0.3)
plt.plot(y, y, 'b')
plt.title("batch_gradient_descent")
plt.show()

y_pred_full = m_full.linear_pred(X)
plt.figure(figsize=[12, 8])
plt.scatter(y_pred_full, y, c='gray', alpha=0.3)
plt.plot(y, y, 'b')
plt.title("full_gradient_descent")
plt.show()

print("stochastic: ", stoch_errors_by_iter[-1]/1e10)
print("batch: ", batch_errors_by_iter[-1]/1e10)
print("full: ", full_errors_by_iter[-1]/1e10)

# error descending
plt.figure(figsize=[4, 6])
plt.subplot(3, 1, 1)
plt.plot(stoch_errors_by_iter)
plt.title("stoch")
plt.subplot(3, 1, 2)
plt.plot(batch_errors_by_iter)
plt.title("batch")
plt.subplot(3, 1, 3)
plt.plot(full_errors_by_iter)
plt.title("full")
plt.tight_layout()
plt.show()
