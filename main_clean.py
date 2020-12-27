#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd # reading .csv files
import numpy as np  # array/image operations


# In[33]:


# load data
train_data = pd.read_csv('dataset/sign_mnist_train.csv')
test_data = pd.read_csv('dataset/sign_mnist_test.csv')
train_data.head()


# In[34]:


# save labels from training data
train_labels = train_data['label'].values

# drop the 'label' column
train_data.drop('label', axis=1, inplace=True)
train_data.head()


# In[35]:


images = train_data.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])


# In[36]:


# convert labels to categorical values
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
train_labels


# In[37]:


# split data into 70% training and 30% testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images,
                                                    train_labels,
                                                    test_size=0.3,
                                                    random_state=101)


# In[40]:


# normalize data
x_train = x_train / 255
x_test = x_test / 255


# In[41]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[42]:


# create the model
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential([
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(rate=0.20),

    Dense(units=24, activation='softmax')
])


# In[43]:


# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[44]:


# train model
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=50,
                    batch_size=128)


# In[47]:


# plot accuracy over epochs
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Over Epochs")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])

plt.show()

