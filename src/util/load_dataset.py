import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def load_dataset(path):
    data = pd.read_csv(path)

    # take only label column
    y = data['label']
    # categorize / binarize labels
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)

    # drop the 'label' column
    x = data.drop(labels='label', axis=1)
    # get the underlying ndarray
    x = x.values
    # normalize the pixel values
    x = x / 255.0
    # reshape them to be 28x28x1 image-like
    x = x.reshape(-1, 28, 28, 1)

    return data, x, y
