from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from model import Model, ClassificationModel, ProbabilityClassificationModel


class IMetric(ABC):
    def __init__(
            self,
            model: Model
    ):
        self._model = model

    @abstractmethod
    def __call__(self, data: pd.DataFrame | pd.Series) -> float | np.ndarray:
        pass


class Accuracy(IMetric):
    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        prediction = self._model(data)
        outcome = data[self._model.outcome]
        t = len(data[~(prediction ^ outcome)])
        return t / len(outcome)


class Precision(IMetric):
    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        prediction = self._model(data)
        outcome = data[self._model.outcome]
        tp = prediction & outcome
        if not any(tp):
            return 0
        return len(data[tp]) / len(data[prediction])


class Recall(IMetric):
    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        prediction = self._model(data)
        outcome = data[self._model.outcome]
        tp = prediction & outcome
        if not any(tp):
            return 0
        return len(data[tp]) / len(data[outcome])


class F1measure(IMetric):
    def __call__(self, data: pd.DataFrame | pd.Series) -> float:
        prediction = self._model(data)
        outcome = data[self._model.outcome]
        tp = (prediction & outcome).sum()
        fp = outcome.sum() - tp
        fn = (~prediction & outcome).sum()
        return tp / (tp + (fp + fn) / 2)


class ROCCurve(IMetric):
    def __init__(
            self,
            model: ProbabilityClassificationModel,
            thresholds: np.ndarray
    ):
        super().__init__(model)
        self.thresholds = thresholds

    def auc(self, data: pd.DataFrame | pd.Series) -> float:
        tpr = []

        outcome = data[self._model.outcome]

        p = outcome.sum()
        if p == 0:
            return 0.0

        for th in self.thresholds:
            prediction = self._model.predict_probability(data) >= th
            tp = (prediction & outcome).sum()
            tpr.append(tp / p)

        return np.mean(tpr, dtype=float)

    def __call__(self, data: pd.DataFrame | pd.Series) -> np.ndarray:
        tpr = []
        fpr = []

        outcome = data[self._model.outcome]
        p = outcome.sum()
        f = len(outcome) - p

        old_threshold = self._model.threshold
        for th in self.thresholds:
            probabilities = self._model.predict_probability(data)
            prediction = probabilities >= th

            tp = (prediction & outcome).sum()
            fp = (prediction & ~outcome).sum()

            tpr.append(tp / p if p != 0 else 0)
            fpr.append(fp / f if f != 0 else 0)

        self._model.threshold = old_threshold
        return np.array([fpr, tpr])
