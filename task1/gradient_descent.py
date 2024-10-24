import pandas as pd

from model import Model
from error import Error, MeanSquareError


def gradient_descent_full(
        model: Model,
        dataset: pd.DataFrame,
        *,
        error_type=MeanSquareError,
        rate=0.01,
        stop=lambda error: error < 0.2,
        max_iteration=1000
):
    error: Error = error_type()

    model.fit()
    error.fit(model)
    i = 0
    while not stop(e := error(dataset)):
        gradient = (error.derivative(dataset) * model.gradient(dataset)).mean(axis=1)
        model.parameter -= gradient * rate
        i += 1
        if i > max_iteration:
            break

    return i, e


def gradient_descent_stochastic(
        model: Model,
        dataset: pd.DataFrame,
        *,
        error_type=MeanSquareError,
        rate=0.01,
        stop=lambda error: error < 0.2,
        max_iteration=1000
):
    error: Error = error_type()

    model.fit()
    error.fit(model)
    i = 0
    while not stop(e := error(dataset)):
        for _, row in dataset.iterrows():
            gradient = error.derivative(row) * model.gradient(row)
            model.parameter -= gradient * rate
        dataset = dataset.sample(frac=1)
        i += 1
        if i > max_iteration:
            break

    return i, e


def gradient_descent_batch(
        model: Model,
        dataset: pd.DataFrame,
        *,
        error_type=MeanSquareError,
        rate=0.01,
        stop=lambda error: error < 0.2,
        max_iteration=1000,
        batch_size=10
):
    error: Error = error_type()

    model.fit()
    error.fit(model)
    i = 0
    while not stop(e := error(dataset)):
        for batch in (dataset[i:i + batch_size if i + batch_size < len(dataset) else len(dataset)]
                      for i in range(0, len(dataset), batch_size)):
            model.parameter -= (error.derivative(batch) * model.gradient(batch)).mean(axis=1) * rate
        dataset = dataset.sample(frac=1)
        i += 1
        if i > max_iteration:
            break

    return i, e
