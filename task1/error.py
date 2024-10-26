from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd

from model import Model


class Error(ABC):
    def __init__(self):
        self._error = None
        self._loss_derivative = None

    @abstractmethod
    def _get_error(self, model: Model) -> Callable[[pd.DataFrame | pd.Series], float]:
        pass

    @abstractmethod
    def _get_loss_derivative(self, model: Model) -> Callable[[pd.DataFrame | pd.Series], np.ndarray]:
        pass

    def fit(self, model: Model) -> None:
        self._error = self._get_error(model)
        self._loss_derivative = self._get_loss_derivative(model)

    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return self._loss_derivative(data)

    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        return self._error(data)


class MeanSquareError(Error):
    def _get_error(self, model: Model) -> Callable[[pd.DataFrame | pd.Series], float]:
        def error(data: pd.DataFrame | pd.Series) -> float:
            return np.sum((model(data) - data[model.outcome]) ** 2).mean()
        return error

    def _get_loss_derivative(self, model: Model) -> Callable[[pd.DataFrame | pd.Series], np.ndarray]:
        def derivative(data: pd.DataFrame | pd.Series) -> np.ndarray:
            return 2 * np.asarray(model(data) - data[model.outcome])
        return derivative

    def score(self, test: pd.DataFrame, outcome: str) -> float:
        return 1. - self(test) / ((test[outcome] - test[outcome].mean()) ** 2).sum()
