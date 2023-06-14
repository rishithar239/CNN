import tensorflow.keras as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = tf.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = dataset.load_data()

#normalization (pre-processing the data)
xtrain = tf.utils.normalize(xtrain, axis=1)
xtest = tf.utils.normalize(xtest, axis=1)
# resizing the image to make it suitable for applying co
IMG_SIZE = 28

#increasing one dimension for kernal operstion
xtrain_r = np.array(xtrain).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
xtest_r = np.array(xtest).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(xtrain_r.shape)
print(xtest_r.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#--------------------------------------------------------------------------------------------#

# creating a neural network
model = Sequential()

#first convolutional layer
model.add(Conv2D(64, (3,3), input_shape = xtrain_r.shape[1:])) #only 1st conv layer mention input layer size
model.add(Activation("relu"))  # to make it non-linear
model.add(MaxPooling2D(pool_size = (2,2) ))

## 2nd convolutional layer
model.add(Conv2D(64, (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size= (2,2) ))

## 3rd convolutional layer
model.add(Conv2D(64, (3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size= (2,2) ))

### fully connected layer 1
model.add(Flatten()) #need to be flattened from 2D to 1D
model.add(Dense(64))
model.add(Activation("relu"))

### layer2
model.add(Dense(32))
model.add(Activation("relu"))

###last fully connected layer(output must be equal to no of classes)
model.add(Dense(10))
model.add(Activation('softmax')) # last-layer actn func must be softmax
#(and sigmoid for binary classification)

model.summary()
model.compile(loss = "sparse_categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])

## training the model
model.fit(xtrain_r, ytrain, epochs=10, validation_split=0.3)

predicted_vals = model.predict(xtest_r)

#testing with an example
print(np.argmax(predicted_vals[0]))
plt.imshow(xtest_r[0], cmap='gray')
plt.show()
