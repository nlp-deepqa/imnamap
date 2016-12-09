import numpy as np


def evaluate_hits(batch_predictions, batch_answers):
    hits = np.zeros(batch_predictions.shape[0])
    for i, predictions in enumerate(batch_predictions):
        hits[i] = sum([1 for prediction in predictions if prediction in batch_answers[i]])
    return np.mean(hits)
