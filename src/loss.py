from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
        pass


class MeanSquaredError(Loss):

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        return 2 * (y_pred - y_true)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sum((y_pred - y_true) ** 2)

class LogLoss(Loss):

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_pred - y_true

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        epsilon = 1e-9
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) *np.log(1 - y_pred + epsilon))