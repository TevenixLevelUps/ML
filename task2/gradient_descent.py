import pandas as pd

from model import Model
from loss import ILoss, MeanSquareError


def get_stop(limit=0.01):
    previous = 0

    def stop(error):
        nonlocal previous
        res = abs(error) < limit or abs(error - previous) < limit
        previous = error
        return res

    return stop


def gradient_descent(
        model: Model,
        dataset: pd.DataFrame,
        batch_size=-1,
        *,
        loss_type=MeanSquareError,
        rate=0.001,
        stop=get_stop(),
        max_iteration=1000,
):
    loss: ILoss = loss_type(model)

    if batch_size == -1 or batch_size > len(dataset):
        batch_size = len(dataset)

    iteration = 0
    while not stop(error := loss(dataset)):
        row = dataset.sample(n=batch_size)
        model.parameter -= loss.gradient(row) * rate
        iteration += 1
        if iteration >= max_iteration:
            break

    return iteration, error
