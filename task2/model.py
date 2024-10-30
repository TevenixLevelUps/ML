from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Model(ABC):
    def __init__(
            self,
            features: list[str],
            outcome: str,
            parameter: np.ndarray,
    ):
        self._features = features
        self._outcome = outcome
        self._parameter = parameter

    @property
    def features(self):
        return self._features.copy()

    @property
    def outcome(self):
        return self._outcome

    @property
    def parameter(self):
        return self._parameter.copy()

    @parameter.setter
    def parameter(self, value: np.ndarray):
        self._parameter = value.copy()

    @abstractmethod
    def gradient(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        pass

    def __call__(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict(data)


class LinearRegressionModel(Model):
    def __init__(
            self,
            features: list[str],
            outcome: str,
    ):
        super().__init__(
            features,
            outcome,
            np.ones_like(features, dtype=np.float64)
        )

    def gradient(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return data[self._features].to_numpy(dtype=np.float64).transpose()

    def predict(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return np.dot(data[self._features], self.parameter)


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
        self.degree = degree

    def predict(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        d = np.asarray(data[self._features])
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        powers = d ** np.arange(self.degree + 1).reshape(-1, 1, 1)
        sums = powers.sum(axis=2).transpose()
        prod = np.dot(sums, self.parameter)

        return prod

    def gradient(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        d = np.asarray(data[self._features])
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        powers = d ** np.arange(self.degree + 1).reshape(-1, 1, 1)

        return powers.sum(axis=2)[..., ::-1]


class ClassificationModel(Model, ABC):
    def __init__(
            self,
            features: list[str],
            outcome: str,
            parameter: np.ndarray,
    ):
        super().__init__(
            features,
            outcome,
            parameter
        )

    @abstractmethod
    def predict(self, data: pd.DataFrame | pd.Series) -> np.ndarray[bool]:
        pass


class ProbabilityClassificationModel(ClassificationModel, ABC):
    def __init__(
            self,
            features: list[str],
            outcome: str,
            parameter: np.ndarray,
            threshold: float = 0.5
    ):
        super().__init__(
            features,
            outcome,
            parameter
        )
        self.threshold = threshold

    def predict(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return self.predict_probability(data) >= self.threshold

    @abstractmethod
    def predict_probability(self, data: pd.DataFrame | pd.Series) -> np.ndarray[bool]:
        pass


class BinaryLogisticRegressionModel(ProbabilityClassificationModel):
    def __init__(
            self,
            model: Model,
            threshold: float = 0.5
    ):
        super().__init__(
            model._features,
            model._outcome,
            model._parameter,
            threshold
        )
        self._model = model

    def predict_probability(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return 1 / (1 + np.exp(-self._model.predict(data)))

    def gradient(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        return self._model.gradient(data)
