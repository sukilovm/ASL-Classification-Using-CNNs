import matplotlib.pyplot as plt


def plot_accuracy(history, save_figure=False, draw_white=False):
    """Plots training and testing accuracy.

    Parameters
    ----------
    history : History
        History object returned by model.fit.
    save_figure : bool
        Whether or not to save figure.
    draw_white : bool
        Whether or not to set text color to white.
    """
    figure = plt.figure(figsize=(15, 5))

    # save property handler
    title_obj = plt.title('Training & Validation Accuracy')

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
                    transparent=True)
    plt.show()
