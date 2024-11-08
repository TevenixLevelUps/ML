import numpy as np
from toolz.itertoolz import getter

from loss import Loss

class LinearRegression:
    def __init__(self,
                 stop: np.float64 = 0.001,
                 learning_rate: np.float64 = 0.01,
                 batch_size: int = -1,
                 epochs: int = 10000) -> None:
        self.stop = stop
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._epochs = epochs
        self._weights = None

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs: int) -> None:
        self._epochs = epochs

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._weights = weights

    def predict(self, data_x: np.ndarray) -> np.ndarray:
        return data_x @ self.weights

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, loss_type: type[Loss], *, weights: np.ndarray = None):
        if weights is None:
            self.weights = np.ones(train_x.shape[1])
        self._gradient_descent(train_x, train_y, loss_type)

    def _gradient_descent(self, data_x: np.ndarray, data_y: np.ndarray, loss_type: type[Loss]):
        if self.batch_size == -1:
            self.batch_size = len(data_x)
        index_x_y = {i: (data_x[i], data_y[i]) for i in range(data_x.shape[0])}
        for _ in range(self.epochs):
            loss = loss_type()

            batch_indexes = np.random.choice(np.array(list(index_x_y.keys())), self.batch_size, replace=False)
            batch_x = np.array([index_x_y[i][0] for i in batch_indexes])
            batch_y = np.array([index_x_y[i][1] for i in batch_indexes])

            y_pred = self.predict(batch_x)
            if loss(data_y, self.predict(data_x)) <= self.stop: break
            gradient = loss.derivative(batch_y, y_pred) * batch_x.transpose()
            self.weights -= self.learning_rate * gradient.mean(axis=1)
