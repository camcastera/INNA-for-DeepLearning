# INNA for Keras

This is the keras implementation for the INNA algorithm based on the paper *An Inertial Newton Algorithm for Deep Learning* ([JMLR version](https://jmlr.csail.mit.edu/papers/v22/19-1024.html)) by C. Castera, J. Bolte, C. Fevotte and E. Pauwels.
It has been tested with Keras 2.2.4 and Tensorflow 1.12.0 as backend, this version is not updated anymore. 

The Tensorflow version, can be found [here](https://github.com/camcastera/INNA-for-DeepLearning/tree/master/INNA_for_tensorflow), a Pytorch implementation is also [available](https://github.com/camcastera/INNA-for-DeepLearning/tree/master/INNA_for_pytorch).

The main code is in the file [inna.py](https://github.com/camcastera/INNA-for-DeepLearning/blob/master/inna_for_keras/inna.py).
## The INNA optimizer can be simply use in the following way:

To use it like any other optimizer (SGD, Adam, Adagrad, etc...), simply do:

```python
# assuming that the file inna.py is in the current folder
from inna import *
```
Then, you will want to compile a model with the optimizer:
```python
inna = INNA(lr=0.01,alpha=0.5,beta=0.1)
model.compile(optimizer=inna)
```

## Below is a complete example on a toy model. 

You can also find it in the file [toy_example.py](https://github.com/camcastera/INNA-for-DeepLearning/blob/master/inna_for_keras/toy_example.py).

```python
# Essential packages

import numpy as np
np.random.seed(27199925)
import keras
from keras import backend as K


# Import the optimizer
from inna import *
inna = INNA(lr=0.5,alpha=0.5,beta=0.1,decay=1.,decaypower=1./2)

# DATASET:

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
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

plt.plot(loss)
plt.show()
```
