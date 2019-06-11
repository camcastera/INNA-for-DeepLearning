# Indian code for Keras

This is the keras implementation for the INDIAN algorithm based on the paper *An Inertial Newton Algorithm for Deep Learning* ([arXiv version](https://arxiv.org/abs/1905.12278)) by C. Castera, J. Bolte, C. Fevotte and E. Pauwels.
It has been tested with Keras This is a Keras 2.2.4 with Tensorflow 1.12.0 as backend. 

To learn how to install and use Keras and Tensorflow, please see [the Keras official website](https://keras.io/).

The main code is in file [indian.py](https://github.com/camcastera/Indian-for-DeepLearning/blob/master/indian_for_keras/indian.py).
## Here is a short example of utilization assuming you have already creating a keras model named model:
To use it like any other optimizer (SGD, Adam, Adagrad, etc...), simply do:

```python
# assuming that the file indian.py is in the current folder
from indian import *
```
 Then when you need to compile a model with this optimizer do:
```python
indian = Indian(lr=0.01,alpha=0.5,beta=0.1,speed_ini=1.,decay=1.,decaypower=0.5)
model.compile(optimizer=indian)
```

## Below there is a more complete example on how to train a toy model with keras. 
You can also find it in the file [toy_example.py](https://github.com/camcastera/Indian-for-DeepLearning/blob/master/indian_for_keras/toy_example.py).

```python
# Essential packages

import numpy as np
np.random.seed(27199925)
import keras
from keras import backend as K


# Import the optimizer
from indian import *
indian = Indian(lr=0.5,alpha=0.5,beta=0.1,speed_ini=1.,decay=1.,decaypower=1./4)

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

model.compile(optimizer=indian, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
