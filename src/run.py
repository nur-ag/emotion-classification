import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support


NUM_WORDS = 500000
BATCH_SIZE = 100000
TARGET_COLUMN = 'emotion_category_index'

def fit_partial(X, y, clf, batch_size):
    num_instances = X.shape[0]
    total_batches = (num_instances + batch_size - 1) // batch_size
    classes = y.unique()
    for i in range(total_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_end = min(batch_end, num_instances)
        X_batch = X[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]
        clf.partial_fit(X_batch, y_batch, classes=classes)
    return clf


def predict_partial(X, clf, batch_size):
    num_instances = X.shape[0]
    total_batches = (num_instances + batch_size - 1) // batch_size
    results = []
    for i in range(total_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_end = min(batch_end, num_instances)
        X_batch = X[batch_start:batch_end]
        results.extend(clf.predict(X_batch).flatten())
    return np.asarray(results).reshape(-1, 1)


vent = pd.read_parquet('../preprocessed/vent.parquet')
frequent_values = vent[TARGET_COLUMN].value_counts()[:9].index
vent_frequent = vent #vent[vent[TARGET_COLUMN].isin(frequent_values)]
task_data = vent_frequent[['text', TARGET_COLUMN]].dropna()
print('Data loading ready...')

rest, test = train_test_split(task_data, test_size=0.2)
train, valid = train_test_split(rest, test_size=0.25)
del vent, rest
print('Splits are ready...')

vec = TfidfVectorizer(stop_words='english', max_features=NUM_WORDS)
X_train = vec.fit_transform(train.text)
X_test = vec.transform(test.text)
print('TF-IDF is ready...')

clf = MultinomialNB()
clf = fit_partial(X_train, train[TARGET_COLUMN], clf, BATCH_SIZE)
print('Training is done...')

y_p_train = predict_partial(X_train, clf, BATCH_SIZE)
y_p_test = predict_partial(X_test, clf, BATCH_SIZE)
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
    