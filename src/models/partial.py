import numpy as np

from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


def reformat_tensor(tensor):
    '''Convert tensors to numpy arrays and shape as a B x d matrix'''
    try:
        tensor = tensor.numpy()
    except:
        tensor = np.asarray(tensor, dtype=np.float32)
    batch_size = len(tensor)
    return tensor.reshape(batch_size, -1)


class SKLearnPartialFitClassifier:
    def __init__(self):
        self.clf = None

    def fit(self, X, y):
        classes = np.unique([y_batch for y_batch in y])
        classes.sort()
        total_batches = min(len(X), len(y))
        for X_batch, y_batch in tqdm(zip(X, y), total=total_batches, desc='Fit'):
            X_batch = reformat_tensor(X_batch)
            y_batch = reformat_tensor(y_batch)
            self.clf.partial_fit(X_batch, y_batch, classes=classes)
        return self

    def predict(self, X):
        results = []
        for X_batch, y_batch in tqdm(X, desc='Predict'):
            X_batch = reformat_tensor(X_batch)
            results.extend(self.clf.predict(X_batch).flatten())
        return np.asarray(results).reshape(-1, 1)


class MultinomialNBPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, **kwargs):
        self.clf = MultinomialNB(**kwargs)


class SGDPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, **kwargs):
        self.clf = SGDClassifier(**kwargs)


class MLPPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, **kwargs):
        self.clf = MLPClassifier(**kwargs)

