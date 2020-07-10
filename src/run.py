import os 
import logging

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from models import MultinomialNBPartialClassifier, SGDPartialClassifier, MLPPartialClassifier, DNNPoolClassifier, LSTMClassifier
from text import TfidfExtractor, FastTextDocEmbeddingExtractor, FastTextTokenEmbeddingExtractor, BertEmbeddingExtractor
from utils.generators import SizedCallableWrapper, SizedBatchWrapper
from utils.split import random_splits

LOGGER = logging.getLogger('EMOTION')

NUM_WORDS = 50000
BATCH_SIZE = 128
TARGET_COLUMN = 'emotion_category_index'
MODEL = DNNPoolClassifier

RAW_PATH = '../preprocessed/vent.parquet'
CACHE_PATH = '../preprocessed/split-cache/'

SPLIT_NAMES = ('train', 'valid', 'test')
SPLIT_PORTIONS = (0.02, 0.01, 0.01)

FASTTEXT_MODEL_PATH = '../cc.en.300.bin'
BERT_MODEL = 'bert-base-uncased'

EXTRACTOR_TYPE = 'bert'
EXTRACTOR_KWARGS = {'ft_model_path': FASTTEXT_MODEL_PATH, 'bert_model': BERT_MODEL, 'num_words': NUM_WORDS}


def load_dataset(raw_path):
    return pd.read_parquet(raw_path)


def preprocess_dataset(dataset):
    frequent_values = dataset[TARGET_COLUMN].value_counts()[:9].index
    frequent = dataset
    frequent = dataset[dataset[TARGET_COLUMN].isin(frequent_values)]
    task_data = frequent[['text', TARGET_COLUMN]].dropna()

    # Remap the classes as necessary to be 0-indexed
    classes = np.unique(task_data[TARGET_COLUMN].to_numpy())
    class_mapping = {c: i for i, c in enumerate(classes)}
    task_data[TARGET_COLUMN] = task_data[TARGET_COLUMN].apply(lambda x: class_mapping[x])
    return task_data


def save_dataset(dataset, path):
    dataset.to_parquet(path)


def load_splits(raw_path=RAW_PATH, 
                cache_path=CACHE_PATH, 
                split_names=SPLIT_NAMES, 
                split_portions=SPLIT_PORTIONS,
                save_cache=True,
                ignore_cache=False):
    # Load the cache if it is available
    if os.path.exists(cache_path) and not ignore_cache:
        try:
            splits = []
            portions_string = '-'.join([str(pt * 100)[:5] for pt in split_portions])
            for split in split_names:
                split_path = '{}{}-{}.parquet'.format(cache_path, split, portions_string)
                splits.append(load_dataset(split_path))
            return splits
        except OSError:
            LOGGER.debug('Could not load cache, loading and preprocessing from raw...')

    # Otherwise load the full dataset and split it
    full_dataset = load_dataset(raw_path)
    clean_dataset = preprocess_dataset(full_dataset)
    all_splits = random_splits(clean_dataset, split_portions)

    # Dump the cache
    if save_cache:
        portions_string = '-'.join([str(pt * 100)[:5] for pt in split_portions])
        for split, dataset in zip(split_names, all_splits):
            split_path = '{}{}-{}.parquet'.format(cache_path, split, portions_string)
            save_dataset(dataset, split_path)

    return all_splits


def prepare_extractor(extractor_type, dataset=None, **kwargs):
    if extractor_type == 'fasttext':
        return FastTextTokenEmbeddingExtractor(kwargs['ft_model_path'])
    if extractor_type == 'tfidf':
        return TfidfExtractor(dataset.text, num_words=kwargs['num_words'])
    if extractor_type == 'bert':
        return BertEmbeddingExtractor(kwargs['bert_model'])
    return None


train, valid, test = load_splits()
print('Splits are ready...')

extractor = prepare_extractor(EXTRACTOR_TYPE, train=train, **EXTRACTOR_KWARGS)
X_train_batch = SizedBatchWrapper(train.text, BATCH_SIZE)
y_train_batch = SizedBatchWrapper(train[TARGET_COLUMN].to_numpy(), BATCH_SIZE)
X_test_batch = SizedBatchWrapper(test.text, BATCH_SIZE)
y_test_batch = SizedBatchWrapper(test[TARGET_COLUMN].to_numpy(), BATCH_SIZE)

X_train = SizedCallableWrapper(X_train_batch, extractor)
X_test = SizedCallableWrapper(X_test_batch, extractor)
print('Feature extractor is ready...')

criterion = torch.nn.CrossEntropyLoss()
clf = MODEL(criterion, input_size=extractor.vector_length())
clf.fit(X_train, y_train_batch)
print('Training is done...')

y_p_train = clf.predict(X_train)
y_p_test = clf.predict(X_test)
print('Predictions are done...')

print('Training data performance: ')
stats = precision_recall_fscore_support(train[TARGET_COLUMN].to_numpy(), y_p_train)
for (p, r, f, s) in zip(*stats):
    print('Precision: {}\tRecall: {}\tF1-score: {}\tSupport: {}'.format(p, r, f, s))
print('Mean macro F1-score: {}'.format(np.mean(stats[2])))
print('Mean micro F1-score: {}'.format(np.sum(stats[2] * stats[3]) / stats[3].sum()))

print('Test data performance: ')
stats = precision_recall_fscore_support(test[TARGET_COLUMN].to_numpy(), y_p_test)
for (p, r, f, s) in zip(*stats):
    print('Precision: {}\tRecall: {}\tF1-score: {}\tSupport: {}'.format(p, r, f, s))
print('Mean macro F1-score: {}'.format(np.mean(stats[2])))
print('Mean micro F1-score: {}'.format(np.sum(stats[2] * stats[3]) / stats[3].sum()))
    
