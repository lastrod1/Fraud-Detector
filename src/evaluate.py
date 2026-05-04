import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def print_metrics(model_name, y_test, y_pred, y_prob):
    print("\nresults for:", model_name)

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print("ROC-AUC Score:", auc)
    print("PR-AUC Score:", pr_auc)


def plot_confusion_matrix(model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    # print it like a table instead of a plot
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix for:", model_name)
    print("true negatives (correctly said NOT fraud):", tn)
    print("false positives (wrongly flagged as fraud):", fp)
    print("false negatives (missed actual fraud!):", fn)
    print("true positives (correctly caught fraud):", tp)