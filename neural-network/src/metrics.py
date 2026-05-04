"""
Evaluation metrics for neural network.
"""

import numpy as np


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Multi-class: argmax
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    else:
        # Binary: threshold at 0.5
        y_pred_class = (y_pred > 0.5).astype(int)
        y_true_class = y_true.astype(int)
    
    return np.mean(y_pred_class == y_true_class)


def precision(y_true, y_pred, average='macro'):
    """Compute precision score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        classes = np.unique(np.concatenate([y_true_class, y_pred_class]))
        
        precisions = []
        for c in classes:
            tp = np.sum((y_pred_class == c) & (y_true_class == c))
            fp = np.sum((y_pred_class == c) & (y_true_class != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(prec)
        
        if average == 'macro':
            return np.mean(precisions)
        else:
            return precisions
    else:
        y_pred_class = (y_pred > 0.5).astype(int)
        y_true_class = y_true.astype(int)
        tp = np.sum((y_pred_class == 1) & (y_true_class == 1))
        fp = np.sum((y_pred_class == 1) & (y_true_class == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true, y_pred, average='macro'):
    """Compute recall score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        classes = np.unique(np.concatenate([y_true_class, y_pred_class]))
        
        recalls = []
        for c in classes:
            tp = np.sum((y_pred_class == c) & (y_true_class == c))
            fn = np.sum((y_pred_class != c) & (y_true_class == c))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(rec)
        
        if average == 'macro':
            return np.mean(recalls)
        else:
            return recalls
    else:
        y_pred_class = (y_pred > 0.5).astype(int)
        y_true_class = y_true.astype(int)
        tp = np.sum((y_pred_class == 1) & (y_true_class == 1))
        fn = np.sum((y_pred_class == 0) & (y_true_class == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true, y_pred, average='macro'):
    """Compute F1 score (harmonic mean of precision and recall)."""
    prec = precision(y_true, y_pred, average=average)
    rec = recall(y_true, y_pred, average=average)
    
    if isinstance(prec, list):
        f1_scores = []
        for p, r in zip(prec, rec):
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
        return np.mean(f1_scores) if average == 'macro' else f1_scores
    else:
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def r2_score(y_true, y_pred):
    """Compute R² score for regression."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    else:
        y_pred_class = (y_pred > 0.5).astype(int)
        y_true_class = y_true.astype(int)
    
    classes = np.unique(np.concatenate([y_true_class, y_pred_class]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            cm[i, j] = np.sum((y_true_class == true_c) & (y_pred_class == pred_c))
    
    return cm
