from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from loss import ILoss
from model import Model


class Regularization(ILoss, ABC):
    def __init__(
            self,
            model: Model,
            loss: ILoss,
            parameter: float
    ):
        super().__init__(model)
        self.parameter = parameter
        self._loss = loss

    def gradient(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return self._loss.gradient(data) + self.parameter * self._regularizer_gradient()

    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return self._loss.derivative(data)

    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        return self._loss(data) + self.parameter * self._regularizer()

    @abstractmethod
    def _regularizer(self) -> float:
        pass

    @abstractmethod
    def _regularizer_gradient(self) -> np.ndarray:
        pass


class RidgeRegularization(Regularization):
    def _regularizer(self) -> float:
        return np.square(self._model.parameter).sum()

    def _regularizer_gradient(self) -> np.ndarray:
        return 2 * self._model.parameter


class LassoRegularization(Regularization):
    def _regularizer(self) -> float:
        return np.abs(self._model.parameter).sum()

    def _regularizer_gradient(self) -> np.ndarray:
        return np.sign(self._model.parameter)
