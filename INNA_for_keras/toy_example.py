####################  Example on a toy problem ####################

# Essential packages

import numpy as np
import keras
from keras import backend as K


# Import the optimizer
from inna import *
inna = INNA(lr=0.1,alpha=0.5,beta=0.1,decay=1.,decaypower=1./2)

# DATASET:

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

# Toy Network:

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = 'relu', 
                 input_shape = (32,32,3)))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Conv2D(filters = 16, 
                 kernel_size = 5,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (14,14,6)))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
model.add(Dense(units = 120, activation = 'relu'))
model.add(Dense(units = 84, activation = 'relu'))

#Output Layer
model.add(Dense(units = 10, activation = 'softmax'))

# Compile the model with the optimizer:

model.compile(optimizer=inna, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Network:
epochs = 10 ; batchsize = 32
HIST = model.fit(x_train, y_train,
              batch_size=batchsize,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose = 1)
              
# Plot the loss function

loss = HIST.history['loss']

import matplotlib.pyplot as plt

plt.plot(np.log10(loss))
plt.show()
