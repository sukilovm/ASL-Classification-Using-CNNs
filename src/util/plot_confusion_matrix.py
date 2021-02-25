import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels):
    """Plots confusion matrix.

    Parameters
    ---------
    y_true : Any
        Ground truth (correct) target values.
    y_pred : Any
        Estimated targets as returned by a classifier.
    labels : Any = "auto"
        xtick and ytick labels
    """
    confusion_mtx = confusion_matrix(y_true, y_pred, normalize=None)
    _, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(data=confusion_mtx,
                annot=True,
                linewidths=0.01,
                cmap=sns.cubehelix_palette(8),
                fmt='d',
                cbar=False,
                ax=ax,
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
