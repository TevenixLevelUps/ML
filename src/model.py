import numpy as np

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


class LogisticRegression:
    def __init__(self,
                 stop: np.float64 = 0.001,
                 learning_rate: np.float64 = 0.01,
                 regularization: str = None,
                 lambda_: float = 0.0,
                 epochs: int = 10000) -> None:
        """
        Parameters:
        - regularization: None, 'l1', or 'l2'
        - lambda_: Strength of regularization
        """
        self.stop = stop
        self.learning_rate = learning_rate
        self._lambda = lambda_
        self._regularization = regularization
        self._epochs = epochs
        self._weights = None
        self._bias = None

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

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias) -> None:
        self._bias = bias

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, loss_type: type[Loss], *, weights: np.ndarray = None):
        n_samples, n_features = train_x.shape

        if weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0

        for _ in range(self.epochs):
            loss = loss_type()
            y_pred = self._feed_forward(train_x)

            gradient = (1 / n_samples) * train_x.transpose() @ loss.derivative(train_y, y_pred)
            bias_grad = (1 / n_samples) * np.sum(loss.derivative(train_y, y_pred))

            if self._regularization == 'l1':
                gradient += self._lambda * np.sign(self.weights)
            elif self._regularization == 'l2':
                gradient += self._lambda * self.weights

            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * bias_grad

    def predict(self, data_x: np.ndarray, threshold: np.float64 = .5) -> np.ndarray:
        y_pred = self.predict_proba(data_x)
        y_pred_class = [1 if i > threshold else 0 for i in y_pred]
        return np.asarray(y_pred_class)

    def predict_proba(self, data_x: np.ndarray) -> np.ndarray:
        return self._sigmoid(np.dot(data_x, self.weights) + self.bias)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _feed_forward(self, x_pred: np.ndarray) -> np.ndarray:
        return self._sigmoid(np.dot(x_pred, self.weights) + self.bias)




