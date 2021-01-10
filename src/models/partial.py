import numpy as np
from scipy.sparse import vstack as sparse_vstack

from tqdm import tqdm

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from incremental_trees.models.classification.streaming_rfc import StreamingRFC


def reformat_tensor(tensor):
    '''Convert tensors to numpy arrays and shape as a B x d matrix'''
    try:
        tensor = tensor.numpy()
    except:
        try:
            tensor = np.asarray(tensor, dtype=np.float32)
        except:
            pass
    batch_size = tensor.shape[0]
    return tensor.reshape(batch_size, -1)


class SKLearnPartialFitClassifier:
    def __init__(self):
        self.clf = None
        self.multilabel = None
        self.num_epochs = 1

    def get_classes(self, y):
        all_batches = [y_batch for y_batch in y]
        if self.multilabel:
            num_classes = len(all_batches[0][0])
            classes = np.eye(num_classes)

            # Hot-patching OneVsRest binarizer as the implementation doesn't support multilabel partial fit
            self.clf.classes_ = [i for i in range(num_classes)]
            self.clf.label_binarizer_ = LabelBinarizer(sparse_output=True).fit(np.eye(num_classes))
            self.clf.estimators_ = [clone(self.clf.estimator) for _ in range(num_classes)]
        else:
            classes = np.unique(all_batches).astype('int32')
            classes.sort()
        return classes

    def fit(self, X, y):
        classes = self.get_classes(y)
        total_batches = min(len(X), len(y))
        for epoch in tqdm(range(self.num_epochs), desc='Epoch'):
            for X_batch, y_batch in tqdm(zip(X, y), total=total_batches, desc='Fit'):
                X_batch = reformat_tensor(X_batch)
                y_batch = reformat_tensor(y_batch).astype('int32')
                self.clf.partial_fit(X_batch, y_batch, classes=classes)
        return self

    def predict(self, X):
        results = []
        for X_batch in tqdm(X, desc='Predict'):
            X_batch = reformat_tensor(X_batch)
            results.extend(self.clf.predict_proba(X_batch))
        result_batches = np.asarray(results)
        num_outputs = result_batches.shape[-1]
        return result_batches.reshape(-1, num_outputs)


class NBPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, problem_type='multiclass', input_size=None, output_size=None, **kwargs):
        super().__init__()
        self.clf = MultinomialNB(**kwargs)
        self.multilabel = problem_type == 'multilabel'
        if self.multilabel:
            self.clf = OneVsRestClassifier(self.clf)


class SGDPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, problem_type='multiclass', input_size=None, output_size=None, **kwargs):
        super().__init__()
        self.clf = SGDClassifier(**kwargs)
        self.multilabel = problem_type == 'multilabel'
        self.num_epochs = kwargs.get('max_iter', 1)
        if self.multilabel:
            self.clf = OneVsRestClassifier(self.clf)


class RFPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, problem_type='multiclass', input_size=None, output_size=None, **kwargs):
        super().__init__()
        num_epochs = kwargs.get('max_iter', 1)
        if 'max_iter' in kwargs:
            del kwargs['max_iter']
        self.clf = StreamingRFC(**kwargs)
        self.multilabel = problem_type == 'multilabel'
        self.num_epochs = num_epochs
        if self.multilabel:
            self.clf = OneVsRestClassifier(self.clf)


class RandomPartialClassifier(SKLearnPartialFitClassifier):
    def __init__(self, problem_type='multiclass', input_size=None, output_size=None, **kwargs):
        super().__init__()
        self.multilabel = problem_type == 'multilabel'
        self.num_classes = None

    def fit(self, X, y):
        print('Doing nothing -- Dummy classifier')
        all_batches = [y_batch for y_batch in y]
        if self.multilabel:
            self.num_classes = len(all_batches[0][0])
        else:
            self.num_classes = len(np.unique(all_batches))

    def predict(self, X):
        scores = []
        for x in X:
            num_samples = x.shape[0]
            random_scores = np.random.random((num_samples, self.num_classes))
            scores.append(random_scores)
        result_batches = np.vstack(scores)
        return result_batches
