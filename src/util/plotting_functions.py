import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.util.load_dataset import load_dataset


def find_first_of(labels, index):
    for i in range(len(labels)):
        if labels[i][index] == 1:
            return i


def plot_own_datasets(save_figure=False):
    train, x_train, y_train = load_dataset('../../dataset/sign_mnist_train.csv')
    _, x_test_user_0, y_test_user_0 \
        = load_dataset('../../dataset/sign_mnist_test_user_0.csv')
    _, x_test_user_1, y_test_user_1 \
        = load_dataset('../../dataset/sign_mnist_test_user_1.csv')
    _, x_test_user_2, y_test_user_2 \
        = load_dataset('../../dataset/sign_mnist_test_user_2.csv')

    figure, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) \
        = plt.subplots(2, 4, gridspec_kw={'wspace': 0, 'hspace': -.5},
                       sharex=True, sharey=True)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax1.imshow(x_train[find_first_of(y_train, 2)], cmap="gray")
    ax2.imshow(x_test_user_0[find_first_of(y_test_user_0, 2)], cmap="gray")
    ax3.imshow(x_test_user_1[find_first_of(y_test_user_1, 2)], cmap="gray")
    ax4.imshow(x_test_user_2[find_first_of(y_test_user_2, 2)], cmap="gray")

    ax5.imshow(x_train[find_first_of(y_train, 10)], cmap="gray")
    ax6.imshow(x_test_user_0[find_first_of(y_test_user_0, 10)], cmap="gray")
    ax7.imshow(x_test_user_1[find_first_of(y_test_user_1, 10)], cmap="gray")
    ax8.imshow(x_test_user_2[find_first_of(y_test_user_2, 10)], cmap="gray")

    if save_figure:
        plt.savefig('../../plots/dataset_sample.png',
                    dpi=figure.dpi,
                    format='png',
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.show()


def plot_accuracy(history, title, save_figure=False, draw_white=False):
    """Plots training and testing accuracy.

    Parameters
    ----------
    history : History
        History object returned by model.fit.
    title : Any
        Title of the plot.
    save_figure : bool
        Whether or not to save figure.
    draw_white : bool
        Whether or not to set text color to white.
    """
    figure = plt.figure(figsize=(15, 5))

    # save property handler
    title_obj = plt.title(title)

    if draw_white:
        params = {"ytick.color": "w",
                  "xtick.color": "w",
                  "axes.labelcolor": "w",
                  "axes.edgecolor": "w"}
        plt.rcParams.update(params)
        plt.setp(title_obj, color='w')

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Evaluation Accuracy')

    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_xlim([0.1, 10])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    if save_figure:
        plt.savefig('../plots/main_result.svg',
                    dpi=figure.dpi,
                    format='svg',
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, save_figure=False):
    """Plots confusion matrix.

    Parameters
    ---------
    y_true : Any
        Ground truth (correct) target values.
    y_pred : Any
        Estimated targets as returned by a classifier.
    labels : Any = "auto"
        xtick and ytick labels.
    save_figure : bool
        Whether or not to save the plot.
    """
    confusion_mtx = confusion_matrix(y_true, y_pred, normalize=None)
    figure, ax = plt.subplots(figsize=(15, 15))
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
    if save_figure:
        plt.savefig('../plots/confusion_matrix.svg',
                    dpi=figure.dpi,
                    format='svg',
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.show()


def plot_countplot(data, save_figure=False):
    figure, ax = plt.subplots()
    sns.countplot(data['label'])
    plt.tight_layout()
    if save_figure:
        plt.savefig('../../plots/countplot.svg',
                    dpi=figure.dpi,
                    format='svg',
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.show()


if __name__ == '__main__':
    train, _, _ = load_dataset('../../dataset/sign_mnist_train.csv')
    plot_own_datasets(save_figure=True)
    plot_countplot(train, save_figure=False)
