from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd


class Model(ABC):
    def __init__(
            self,
            features: list[str],
            outcome: str,
            parameter: np.ndarray,
    ):
        self.features = features
        self.outcome = outcome
        self.parameter = parameter
        self._model = None
        self._gradient = None

    @property
    def gradient(self) -> Callable[[pd.DataFrame | pd.Series], np.ndarray]:
        return self._gradient

    @abstractmethod
    def _get_model(self, parameter: np.ndarray) -> Callable[[pd.DataFrame| pd.Series], pd.Series]:
        pass

    @abstractmethod
    def _get_model_gradient(self, parameter: np.ndarray) -> Callable[[pd.DataFrame | pd.Series], np.ndarray]:
        pass

    def fit(self) -> None:
        self._model = self._get_model(self.parameter)
        self._gradient = self._get_model_gradient(self.parameter)

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return self._model(data)

    def __bool__(self):
        return self._model is not None


class LinearRegressionModel(Model):
    def __init__(
            self,
            features: list[str],
            outcome: str,
    ):
        super().__init__(
            features,
            outcome,
            np.ones(len(features) + 1, dtype=np.float64)
        )

    def _get_model(self, parameter: np.ndarray) -> Callable[[pd.DataFrame | pd.Series], pd.Series]:
        args = self.features.copy()

        def model(data: pd.DataFrame| pd.Series) -> pd.Series:
            return np.dot(data[args], parameter[:len(args)]) + parameter[len(args)]

        return model

    def _get_model_gradient(self, parameter: np.ndarray) -> Callable[[pd.DataFrame | pd.Series], np.ndarray]:
        args = self.features.copy()

        def gradient(data: pd.DataFrame | pd.Series) -> pd.Series:
            grad = data[args].copy()
            grad["fff"] = 1
            return grad.to_numpy(dtype=np.float64).transpose()

        return gradient


class PolynomialRegressionModel(Model):
    def __init__(
            self,
            features: list[str],
            outcome: str,
            *,
            degree: int = 2
    ):
        super().__init__(
            features,
            outcome,
            np.ones(degree + 1, dtype=np.float64)
        )

    def _get_model(self, parameter: np.ndarray) -> Callable[[pd.DataFrame | pd.Series], pd.Series]:
        args = self.features.copy()

        def model(data: pd.DataFrame | pd.Series):
            d = data[args]
            if d.ndim == 1:
                d = np.asarray(d, dtype=np.float64).reshape(-1, 1)
            return np.sum(
                np.asarray(
                    [np.asarray(d) ** i * self.parameter[i] for i in range(len(self.parameter))]
                ),
                axis=(0, 2)
            )

        return model

    def _get_model_gradient(self, parameter: np.ndarray) -> Callable[[pd.DataFrame | pd.Series], np.ndarray]:
        args = self.features.copy()

        def gradient(data: pd.DataFrame):
            return np.fromfunction(
                lambda i: np.asarray(data[args], dtype=np.float64) ** i,
                [len(self.parameter)]
            ).transpose()

        return gradient
