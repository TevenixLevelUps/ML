import numpy as np


def accuracy(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    false_positives = np.sum((1 - y_true) * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)


def precision(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum((1 - y_true) * y_pred)
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))
    return true_positives / (true_positives + false_negatives)


def f1_score(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum((1 - y_true) * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return 2 * (precision * recall) / (precision + recall)


def roc_curve(y_true, y_score):
    tpr = [0]
    fpr = [0]
    total_positives = np.sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in np.sort(y_score)[::-1]:
        y_pred = (y_score >= threshold)

        tp = np.sum(y_true * y_pred)
        fp = np.sum((1 - y_true) * y_pred)

        tpr.append(tp / total_positives)
        fpr.append(fp / total_negatives)

    fpr.append(1)
    tpr.append(1)

    return fpr, tpr


def roc_auc_score(y_true, y_score):
    fpr, tpr = roc_curve(y_true, y_score)
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return auc