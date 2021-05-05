import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve


def classification_report(y_true, y_pred, labels):
    stats = precision_recall_fscore_support(y_true, y_pred)
    label_results = {}
    report = {'labels': label_results}
    for (label, values) in zip(labels, zip(*stats)):
        (prec, recall, f1, support) = values
        label_results[label] = {
            'precision': float(prec),
            'recall': float(recall),
            'f1': float(f1),
            'support': float(support)
        }
    report['macro_f1'] = float(np.mean(stats[2]))
    report['micro_f1'] = float(np.sum(stats[2] * stats[3]) / stats[3].sum())
    report['micro_precision'] = float(np.sum(stats[0] * stats[3]) / stats[3].sum())
    report['micro_recall'] = float(np.sum(stats[1] * stats[3]) / stats[3].sum())
    return report


def discretize_output(logits_tensor, problem_type, threshold=0.5):
    if problem_type == 'multiclass':
        return logits_tensor.argmax(axis=-1)
    elif problem_type == 'multilabel':
        return logits_tensor > threshold
    return logits_tensor


def _find_thresholds(logits, y_true):
    thresholds = []
    f1_scores = []
    num_labels = logits.shape[-1]
    for i in range(num_labels):
        p, r, th = precision_recall_curve(y_true[:, i], logits[:, i])
        f1 = np.nan_to_num(2 * p * r / (p + r), 0)
        f1_max = f1.argmax()
        thresholds.append(th[f1_max])
        f1_scores.append(f1[f1_max])
    return np.asarray(thresholds), np.asarray(f1_scores)


def find_thresholds(logits, y_true, batch_size=400000):
    total_elems = len(logits)
    thresholds = None
    f1_scores = None
    for start in range(0, total_elems, batch_size):
        end = start + batch_size
        tmp_ths, tmp_f1s = _find_thresholds(logits[start:end], y_true[start:end])
        if f1_scores is None:
            thresholds = tmp_ths
            f1_scores = tmp_f1s
        else:
            is_higher = tmp_f1s > f1_scores
            np.putmask(thresholds, is_higher, tmp_ths)
            np.putmask(f1_scores, is_higher, tmp_f1s)
    return thresholds, f1_scores
