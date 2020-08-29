import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score


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


def discretize_output(logits_tensor, problem_type, threshold=0.5):
    if problem_type == 'multiclass':
        return logits_tensor.argmax(axis=-1)
    elif problem_type == 'multilabel':
        return logits_tensor > threshold
    return logits_tensor


def find_thresholds(logits, y_true, granularity=0.001, metric=f1_score):
    thresholds = np.arange(0, 1, 0.001)
    num_columns = logits.shape[-1]
    
    # Handle the case in which the vectors are flat
    if len(logits.shape) == 1:
        logits = logits.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        num_columns = 1

    # For every set of columns, identify the optimal threshold
    column_thresholds = []
    column_scores = []
    for column in range(num_columns):
        max_score = 0.0
        best_threshold = 0.0
        logits_vector = logits[:, column]
        truth_vector = y_true[:, column]
        for threshold in thresholds:
            score = metric(truth_vector, logits_vector > threshold)
            if score > max_score:
                max_score = score
                best_threshold = threshold
        column_thresholds.append(best_threshold)
        column_scores.append(max_score)
    return np.asarray(column_thresholds), np.asarray(column_scores)
