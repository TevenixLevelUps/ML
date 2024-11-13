from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from metric import IMetric
from model import ClassificationModel


class ILoss(IMetric, ABC):
    @abstractmethod
    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        pass

    def gradient(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return (self.derivative(data) * self._model.gradient(data)).mean(axis=1)

    @abstractmethod
    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        pass


class MeanSquareError(ILoss):
    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return 2 * np.asarray(self._model(data) - data[self._model.outcome])

    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        return np.sum((self._model(data) - data[self._model.outcome]) ** 2).mean()


class LogLoss(ILoss):
    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        outcome = np.asarray(data[self._model.outcome])
        prediction = self._model.predict(data)
        sigmoid = 1 / (1 + np.exp(-prediction))
        return outcome - sigmoid

    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        outcome = data[self._model.outcome]
        prediction = self._model.predict(data)
        return np.sum(outcome * prediction - np.log1p(np.exp(prediction)))
