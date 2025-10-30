import numpy as np


def accuracy_score(y_true, y_pred, normalize=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    score = np.sum(y_true == y_pred)
    
    if normalize:
        score = score / len(y_true)
    
    return score


def mean_squared_error(y_true, y_pred, squared=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    mse = np.mean((y_true - y_pred) ** 2)
    
    if squared:
        return mse
    else:
        return np.sqrt(mse)


def classification_report(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    class_idx = {label: i for i, label in enumerate(classes)}

    support = np.zeros(n_classes, dtype=int)
    tp = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)

    for i, label in enumerate(classes):
        tp[i] = np.sum((y_pred == label) & (y_true == label))
        fp[i] = np.sum((y_pred == label) & (y_true != label))
        fn[i] = np.sum((y_pred != label) & (y_true == label))
        support[i] = np.sum(y_true == label)

    precision = np.zeros(n_classes, dtype=float)
    recall = np.zeros(n_classes, dtype=float)
    f1 = np.zeros(n_classes, dtype=float)
    for i in range(n_classes):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
        else:
            precision[i] = 0.0
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
        else:
            recall[i] = 0.0
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0.0

    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)

    weights = support / support.sum()
    weighted_p = np.sum(precision * weights)
    weighted_r = np.sum(recall * weights)
    weighted_f1 = np.sum(f1 * weights)

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    print(f"Accuracy: {accuracy:.12f}\n")

    head_fmt = "{:>15} {:>10} {:>10} {:>10} {:>10}"
    row_fmt = "{:>15} {:10.2f} {:10.2f} {:10.2f} {:10d}"
    print(head_fmt.format("", "precision", "recall", "f1-score", "support"))
    print()
    for i, label in enumerate(classes):
        print(
            row_fmt.format(
                str(label),
                precision[i],
                recall[i],
                f1[i],
                support[i],
            )
        )
    print()
    acc_fmt = "{:>15} {:>10} {:>10} {:>10} {:>10}"
    acc_row = "{:>15} {:>10} {:>10} {:>10} {:10d}"
    print(
        acc_row.format(
            "accuracy", "", "", f"{accuracy:.2f}", support.sum()
        )
    )
    print()
    avg_row_fmt = "{:>15} {:10.2f} {:10.2f} {:10.2f} {:10d}"
    print(
        avg_row_fmt.format("macro avg", macro_p, macro_r, macro_f1, support.sum())
    )
    print(
        avg_row_fmt.format("weighted avg", weighted_p, weighted_r, weighted_f1, support.sum())
    )