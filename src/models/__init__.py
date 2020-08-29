import torch
from .partial import NBPartialClassifier, SGDPartialClassifier
from .neural import DNNPoolClassifier, LSTMClassifier


def model_factory(model_name):
    model_name = model_name.lower()
    if model_name == 'naivebayes':
        return NBPartialClassifier
    if model_name == 'sgd':
        return SGDPartialClassifier
    if model_name == 'dnnpool':
        return DNNPoolClassifier
    if model_name == 'lstm':
        return LSTMClassifier
    raise RuntimeError('Unknown model name "{}".'.format(model_name))
