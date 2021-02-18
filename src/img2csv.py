import os
from PIL import Image
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def create_csv_from_image_data(directory_path, csv_name):
    for dir_path, dir_names, filenames in os.walk(directory_path):
        first_row = ['label']
        for pixel_nr in range(1, 28 * 28 + 1):
            first_row.append('pixel{}'.format(str(pixel_nr)))
        with open(csv_name, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(first_row)
        for filename in filenames:
            # print(os.path.splitext(filename)[0])
            img = Image.open(os.path.join(dir_path, filename))
            img = img.resize((28, 28))
            img_grey = img.convert('L')
            value = np.asarray(img_grey.getdata(), dtype=np.int) \
                .reshape((28, 28)) \
                .flatten()
            value = np.insert(value, 0, (ord(filename[0]) - ord('a')))
            with open(csv_name, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(value)


if __name__ == '__main__':
    create_csv_from_image_data('../imgs/mariia', 'sign_mnist_test_mariia.csv')

    csv_file = pd.read_csv('../dataset/sign_mnist_test_mariia.csv')

    labels = csv_file['label']
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    # print(labels)

    data = csv_file.drop(labels='label', axis=1)
    data = (data.values / 255.0).reshape(-1, 28, 28, 1)

    plt.imshow(data[0], cmap="gray")
    plt.show()
