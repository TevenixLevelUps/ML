from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd

from model import Model


class Loss(ABC):
    def __init__(
            self,
            model: Model
    ):
        self._model = model

    @abstractmethod
    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        pass


class MeanSquareError(Loss):
    def derivative(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return 2 * np.asarray(self._model(data) - data[self._model.outcome])

    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        return np.sum((self._model(data) - data[self._model.outcome]) ** 2).mean()
