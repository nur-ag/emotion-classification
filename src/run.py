import os 
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from models import MultinomialNBPartialClassifier, SGDPartialClassifier, MLPPartialClassifier, DNNPoolClassifier, LSTMClassifier
from text import TfidfExtractor, FastTextDocEmbeddingExtractor, FastTextTokenEmbeddingExtractor, BertEmbeddingExtractor
from utils.generators import SizedCallableWrapper, SizedBatchWrapper
from utils.split import random_splits, column_splits
from utils.random import set_seed

from config import DataConfig, ExtractorConfig, ModelConfig, ExperimentConfig

LOGGER = logging.getLogger('EMOTION')

SEED = 123
DATASET = 'goem'
LABEL_NAMES = list(map(str, range(10000))) if DATASET == 'vent' else [l.strip().capitalize() for l in open('../data/GoEmotionsNames.txt')]

SPLIT_NAMES = ('train', 'valid', 'test')
SPLIT_PORTIONS = (0.02, 0.01, 0.01)
PROJECT_ROOT_PATH = str(Path(__file__).parent.absolute().parent)

# Dataset loading configurations
VENT_DATA_CONFIG = DataConfig(raw_path='{}/preprocessed/vent.parquet'.format(PROJECT_ROOT_PATH), 
                              cache_path='{}/preprocessed/vent-split-cache/'.format(PROJECT_ROOT_PATH), 
                              split_names=SPLIT_NAMES, 
                              split_portions=SPLIT_PORTIONS, 
                              split_mode='random', 
                              dataset_format='vent', 
                              target_column='emotion_index', 
                              text_column='text')

GOEM_DATA_CONFIG = DataConfig(raw_path='{}/preprocessed/GoEmotions.parquet'.format(PROJECT_ROOT_PATH), 
                              cache_path='{}/preprocessed/GoEmotions-split-cache/'.format(PROJECT_ROOT_PATH), 
                              split_names=SPLIT_NAMES, 
                              split_portions=SPLIT_PORTIONS, 
                              split_mode='column', 
                              dataset_format='goemotions', 
                              target_column='emotions_label', 
                              text_column='text')

# Extractor configuration
EXTRACTOR_CONFIG = ExtractorConfig(ex_type='bert', ex_args={
                        'ft_model_path': '{}/cc.en.300.bin', 
                        'bert_model': 'bert-base-uncased', 
                        'num_words': 50000,
                        'freeze_output': False,
                        'max_length': 50}
                    )

# Model configuration
VENT_MODEL_CONFIG = ModelConfig(model_name='dnnpool', 
                                problem_type='multiclass', 
                                batch_size=16, 
                                model_conf={})

GOEM_MODEL_CONFIG = ModelConfig(model_name='dnnpool', 
                                problem_type='multilabel', 
                                batch_size=16, 
                                model_conf={})

# Complete model config
EXPERIMENT_CONFIG = ExperimentConfig(data_config=VENT_DATA_CONFIG if DATASET == 'vent' else GOEM_DATA_CONFIG,
                                     extractor_config=EXTRACTOR_CONFIG,
                                     model_config=VENT_MODEL_CONFIG if DATASET == 'vent' else GOEM_MODEL_CONFIG,
                                     label_names=LABEL_NAMES,
                                     seed=SEED)

def load_dataset(raw_path):
    return pd.read_parquet(raw_path)


def preprocess_vent(cleared, target_column):
    filtered = cleared[cleared.enabled & cleared.plain_name]
    classes = np.unique(filtered[target_column].to_numpy())
    class_mapping = {c: i for i, c in enumerate(classes)}
    filtered[target_column] = filtered[target_column].apply(lambda x: class_mapping[x])
    return filtered


def preprocess_dataset(dataset, data_format, target_column, text_column):
    if data_format == 'vent':
        dataset = preprocess_vent(dataset, target_column)
        task_data = dataset[[text_column, target_column]].dropna()
    elif DATASET_FORMAT == 'goemotions':
        task_data = dataset[[text_column, target_column, 'split']].dropna()
    else:
        task_data = dataset[[text_column, target_column]].dropna()
    return task_data


def save_dataset(dataset, path):
    dataset.to_parquet(path)


def load_splits(raw_path, 
                cache_path, 
                split_names, 
                split_portions,
                split_mode,
                dataset_format,
                target_column,
                text_column,
                save_cache=True,
                ignore_cache=False):
    # Load the cache if it is available
    if split_mode == 'column':
        portions_string = 'cols'
    else:
        portions_string = '-'.join([str(pt * 100)[:5] for pt in split_portions])
    if os.path.exists(cache_path) and not ignore_cache:
        try:
            splits = []
            for split in split_names:
                split_path = '{}{}-{}.parquet'.format(cache_path, split, portions_string)
                splits.append(load_dataset(split_path))
            return splits
        except OSError:
            LOGGER.debug('Could not load cache, loading and preprocessing from raw...')

    # Otherwise load the full dataset and split it
    full_dataset = load_dataset(raw_path)
    clean_dataset = preprocess_dataset(full_dataset, dataset_format, target_column, text_column)

    if split_mode == 'random':
        all_splits = random_splits(clean_dataset, split_portions)
    elif split_mode == 'column':
        all_splits = column_splits(clean_dataset)
    else:
        raise RuntimeError('Unknown split mode "{}".'.format(split_mode))

    # Dump the cache
    if save_cache:
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        for split, dataset in zip(split_names, all_splits):
            split_path = '{}{}-{}.parquet'.format(cache_path, split, portions_string)
            save_dataset(dataset, split_path)

    return all_splits


def prepare_extractor(extractor_type, dataset=None, **kwargs):
    if extractor_type == 'fasttext':
        return FastTextTokenEmbeddingExtractor(kwargs['ft_model_path'], 
                                               max_length=kwargs['max_length'])
    if extractor_type == 'tfidf':
        return TfidfExtractor(dataset, num_words=kwargs['num_words'])
    if extractor_type == 'bert':
        return BertEmbeddingExtractor(kwargs['bert_model'], 
                                      freeze_output=kwargs['freeze_output'], 
                                      max_length=kwargs['max_length'])
    return None


def loss_function(problem_type):
    if problem_type == 'multiclass':
        return torch.nn.CrossEntropyLoss()
    elif problem_type == 'multilabel':
        return torch.nn.MultiLabelSoftMarginLoss()
    return None


def model_factory(model_name):
    model_name = model_name.lower()
    if model_name == 'multinb':
        return MultinomialNBPartialClassifier
    if model_name == 'sgd':
        return SGDPartialClassifier
    if model_name == 'mlp': 
        return MLPPartialClassifier
    if model_name == 'dnnpool':
        return DNNPoolClassifier
    if model_name == 'lstm':
        return LSTMClassifier
    raise RuntimeError('Unknown model name "{}".'.format(model_name))


def log_report(report):
    for label, stats in report['labels'].items():
        p = stats['precision']
        r = stats['recall']
        f = stats['f1']
        s = stats['support']
        LOGGER.info('{}\tPrecision: {}\tRecall: {}\tF1-score: {}\tSupport: {}'.format(label, p, r, f, s))
    LOGGER.info('')
    LOGGER.info('Mean macro F1-score: {}'.format(report['macro_f1']))
    LOGGER.info('Mean micro F1-score: {}'.format(report['micro_f1']))


def classification_report(y_true, y_pred, labels):
    stats = precision_recall_fscore_support(y_true, y_pred)
    label_results = {}
    report = {'labels': label_results}
    for (label, values) in zip(labels, zip(*stats)):
        (prec, recall, f1, support) = values
        label_results[label] = {
            'precision': prec,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    report['macro_f1'] = np.mean(stats[2])
    report['micro_f1'] = np.sum(stats[2] * stats[3]) / stats[3].sum()
    return report


def discretize_output(logits_tensor, problem_type):
    if problem_type == 'multiclass':
        return logits_tensor.argmax(axis=-1)
    elif problem_type == 'multilabel':
        return logits_tensor > 0.0
    return logits_tensor


def emotion_experiment(experiment_config):
    data_config = experiment_config.data_config
    extractor_config = experiment_config.extractor_config
    model_config = experiment_config.model_config
    if experiment_config.seed is not None:
        set_seed(experiment_config.seed)
        LOGGER.info('Seeding all libraries with seed "{}"...'.format(experiment_config.seed))

    train, valid, test = load_splits(**data_config._asdict())
    LOGGER.info('Splits are ready...')

    y_train_array = np.asarray(train[data_config.target_column].to_list(), dtype=np.float32)
    y_valid_array = np.asarray(valid[data_config.target_column].to_list(), dtype=np.float32)
    y_test_array = np.asarray(test[data_config.target_column].to_list(), dtype=np.float32)
    num_labels = np.unique(y_train_array).size if len(y_train_array.shape) == 1 else y_train_array.shape[-1]
    LOGGER.info('Labels are ready, with a total number of {} possible targets...'.format(num_labels))

    extractor = prepare_extractor(extractor_config.ex_type, dataset=train[data_config.text_column], **extractor_config.ex_args)
    X_train_batch = SizedBatchWrapper(train[data_config.text_column], model_config.batch_size)
    X_valid_batch = SizedBatchWrapper(valid[data_config.text_column], model_config.batch_size)
    X_test_batch = SizedBatchWrapper(test[data_config.text_column], model_config.batch_size)
    y_train_batch = SizedBatchWrapper(y_train_array, model_config.batch_size)

    X_train = SizedCallableWrapper(X_train_batch, extractor)
    X_valid = SizedCallableWrapper(X_valid_batch, extractor)
    X_test = SizedCallableWrapper(X_test_batch, extractor)
    LOGGER.info('Feature extractor is ready...')

    criterion = loss_function(model_config.problem_type)
    model = model_factory(model_config.model_name)
    clf = model(criterion, input_size=extractor.vector_length(), output_size=num_labels)
    clf.fit(X_train, y_train_batch)
    LOGGER.info('Training is done...')

    y_p_train = discretize_output(clf.predict(X_train), model_config.problem_type)
    y_p_valid = discretize_output(clf.predict(X_valid), model_config.problem_type)
    y_p_test = discretize_output(clf.predict(X_test), model_config.problem_type)
    LOGGER.info('Predictions are done...')

    train_report = classification_report(y_train_array, y_p_train, experiment_config.label_names)
    valid_report = classification_report(y_valid_array, y_p_valid, experiment_config.label_names)
    test_report  = classification_report(y_test_array, y_p_test, experiment_config.label_names)

    results = {}
    for split, report in zip(data_config.split_names, [train_report, valid_report, test_report]):
        results[split] = report
        LOGGER.info('Classification report -- {}'.format(split.capitalize()))
        LOGGER.info('#################################')
        LOGGER.info('')
        log_report(report)
        LOGGER.info('')
        LOGGER.info('')

    return {'config': experiment_config._as_flat_dict(),
            'results': results}


if __name__ == "__main__":
    emotion_experiment(EXPERIMENT_CONFIG)
