import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from error import MeanSquareError
from gradient_descent import gradient_descent_full, gradient_descent_stochastic, gradient_descent_batch
from model import LinearRegressionModel, PolynomialRegressionModel


DATASET_SIZE = 200
feats = ["total_rooms", "total_bedrooms", "housing_median_age"]
outcome = "median_house_value"

dataset = pd.read_csv("housing.csv").head(DATASET_SIZE)

for arg in [*feats, outcome]:
    dataset[arg] = (dataset[arg] - dataset[arg].min()) / (dataset[arg].max() - dataset[arg].min())

test = dataset.tail(DATASET_SIZE // 4)
dataset = dataset.head(3 * DATASET_SIZE // 4)

for feat in feats:
    sns.lineplot(
        x=dataset[feat],
        y=dataset[outcome]
    )
    plt.show()

examples = [
    {
        "model_type": LinearRegressionModel,
        "kwargs":
            {
                "dataset": dataset,
                "stop": lambda error: False,
                "rate": 0.01,
            },
        "model_kwargs":
            {
                "features": feats,
                "outcome": outcome,
            },
        "subtitle": "Линейная регрессия",
    },
    {
        "model_type": PolynomialRegressionModel,
        "kwargs":
            {
                "dataset": dataset,
                "stop": lambda error: False,
                "rate": 0.001,
            },
        "model_kwargs":
            {
                "features": feats,
                "outcome": outcome,
                "degree": 2
            },
        "subtitle": "Полиномиальная регрессия 2 степени",
    },
]

error = MeanSquareError()


def learn_error(num, title, method, subtitle, model_type, model_kwargs, kwargs):
    errors = []
    scores = []
    iterations = []
    for iteration in np.arange(10) * 3:
        model = model_type(**model_kwargs)
        kwargs["max_iteration"] = iteration

        i, _ = method(model, **kwargs)

        error.fit(model)
        errors.append(error(test))
        scores.append(error.score(test, outcome))
        iterations.append(i)

    plt.subplot(1, 3, num)
    sns.pointplot(
        x=iterations,
        y=errors,
    )
    sns.pointplot(
        x=iterations,
        y=scores,
    )
    plt.suptitle(subtitle)
    plt.title(f'График зависимости значеня функции потерь от количества итераций\n{title}')
    plt.xlabel('Количество итераций')
    plt.ylabel('Ошибка на тестовой выборке, R^2')


def learn_accuracy(num, title, method, subtitle, model_type, model_kwargs, kwargs):
    accuracies = []
    iterations = []
    errors = []
    for limit_error in (20 + np.arange(5) * 3):
        kwargs["stop"] = lambda err: err < limit_error
        kwargs["max_iteration"] = 200
        model = model_type(**model_kwargs)

        i, _ = method(model, **kwargs)

        error.fit(model)
        errors.append(error(test))
        iterations.append(i)
        accuracies.append(limit_error)

    plt.subplot(1, 3, num)
    sns.pointplot(
        x=accuracies,
        y=iterations,

    )
    sns.pointplot(
        x=accuracies,
        y=errors,
    )
    plt.suptitle(subtitle)
    plt.title(f'График зависимостиколичества итераций от заданной точности\n{title}')
    plt.xlabel(f'Порог точности')
    plt.ylabel('Количество итераций, ошибка')


for example in examples:
    learn_error(1, "Полный градиентный спуск", gradient_descent_full, **example)
    learn_error(2, "Стохастический градиентный спуск", gradient_descent_stochastic, **example)
    learn_error(3, "Батчевый градиентный спуск", gradient_descent_batch, **example)
    plt.show()

for example in examples:
    learn_accuracy(1, "Полный градиентный спуск", gradient_descent_full, **example)
    learn_accuracy(2, "Стохастический градиентный спуск", gradient_descent_stochastic, **example)
    learn_accuracy(3, "Батчевый градиентный спуск", gradient_descent_batch, **example)
    plt.show()
