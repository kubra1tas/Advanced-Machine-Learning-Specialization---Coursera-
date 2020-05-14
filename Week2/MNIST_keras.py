import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras_utils

import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)
import grading_utils
import keras_utils
from keras_utils import reset_tf_session

# import necessary building blocks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU

model.add(MaxP)

data, labels = tf.keras.datasets.mnist.load_data()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
X_train = np.array(X_train)
X_val = np.array(X_test)
X_train_flat = X_train.reshape((X_train.shape[0], -1))
print(X_train_flat.shape)

X_val_flat = X_val.reshape((X_val.shape[0], -1))
print(X_val_flat.shape)

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

print(y_train_oh.shape)
print(y_train_oh[:3], y_train[:3])


model = Sequential()
model.add(Conv2D(16, (3,3), 'same', 'relu', input_shape = (32,32,3)))
model.add(Conv2D(32, (3,3), 'same', 'relu'))