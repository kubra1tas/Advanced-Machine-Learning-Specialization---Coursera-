
# coding: utf-8

# # Your first CNN on CIFAR-10
# 
# In this task you will: 
# * define your first CNN architecture for CIFAR-10 dataset
# * train it from scratch
# * visualize learnt filters
# 
# CIFAR-10 dataset contains 32x32 color images from 10 classes: __airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck__:
# <img src="images/cifar10.jpg" style="width:80%">

# # Import stuff




# In[2]:

import sys
sys.path.append("..")
import grading
import download_utils

# In[2]:


# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!


# In[4]:


import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)
#import grading_utils
import keras_utils


# # Fill in your Coursera token and email
# To successfully submit your answers to our grader, please fill in your Coursera submission token and email




# In[2]:


# !!! remember to clear session/graph if you rebuild your graph to avoid out-of-memory errors !!!


# In[4]:


import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import keras_utils


# # Fill in your Coursera token and email
# To successfully submit your answers to our grader, please fill in your Coursera submission token and email




# In[ ]:


# token expires every 30 min
#COURSERA_TOKEN = ### YOUR TOKEN HERE
#COURSERA_EMAIL = ### YOUR EMAIL HERE


# # Load dataset

# In[6]:


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_train.shape)

# In[7]:


print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)


# In[8]:


NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]


# In[9]:


# show random images from train
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()


# # Prepare data

# We need to normalize inputs like this: $$x_{norm} = \frac{x}{255} - 0.5$$
# 
# We need to convert class labels to one-hot encoded vectors. Use __keras.utils.to_categorical__.

# In[10]:


# normalize inputs
x_train2 = x_train/255-0.5### YOUR CODE HERE
x_test2 = x_test/255-0.5### YOUR CODE HERE

# convert class labels to one-hot encoded, should have shape (?, NUM_CLASSES)
y_train2 = keras.utils.to_categorical(y_train, num_classes = 10)### YOUR CODE HERE
y_test2 = keras.utils.to_categorical(y_test, num_classes = 10)### YOUR CODE HERE



# # Define CNN architecture

# In[11]:

# import necessary building blocks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU


# Convolutional networks are built from several types of layers:
# - [Conv2D](https://keras.io/layers/convolutional/#conv2d) - performs convolution:
#     - **filters**: number of output channels; 
#     - **kernel_size**: an integer or tuple/list of 2 integers, specifying the width and height of the 2D convolution window;
#     - **padding**: padding="same" adds zero padding to the input, so that the output has the same width and height, padding='valid' performs convolution only in locations where kernel and the input fully overlap;
#     - **activation**: "relu", "tanh", etc.
#     - **input_shape**: shape of input.
# - [MaxPooling2D](https://keras.io/layers/pooling/#maxpooling2d) - performs 2D max pooling.
# - [Flatten](https://keras.io/layers/core/#flatten) - flattens the input, does not affect the batch size.
# - [Dense](https://keras.io/layers/core/#dense) - fully-connected layer.
# - [Activation](https://keras.io/layers/core/#activation) - applies an activation function.
# - [LeakyReLU](https://keras.io/layers/advanced-activations/#leakyrelu) - applies leaky relu activation.
# - [Dropout](https://keras.io/layers/core/#dropout) - applies dropout.

# You need to define a model which takes __(None, 32, 32, 3)__ input and predicts __(None, 10)__ output with probabilities for all classes. __None__ in shapes stands for batch dimension.
# 
# Simple feed-forward networks in Keras can be defined in the following way:
# 
# ```python
# model = Sequential()  # start feed-forward model definition
# model.add(Conv2D(..., input_shape=(32, 32, 3)))  # first layer needs to define "input_shape"
# 
# ...  # here comes a bunch of convolutional, pooling and dropout layers
# 
# model.add(Dense(NUM_CLASSES))  # the last layer with neuron for each class
# model.add(Activation("softmax"))  # output probabilities
# ```
# 
# Stack __4__ convolutional layers with kernel size __(3, 3)__ with growing number of filters __(16, 32, 32, 64)__, use "same" padding.
# 
# Add __2x2__ pooling layer after every 2 convolutional layers (conv-conv-pool scheme).
# 
# Use __LeakyReLU__ activation with recommended parameter __0.1__ for all layers that need it (after convolutional and dense layers):
# ```python
# model.add(LeakyReLU(0.1))
# ```
# 
# Add a dense layer with __256__ neurons and a second dense layer with __10__ neurons for classes. Remember to use __Flatten__ layer before first dense layer to reshape input volume into a flat vector!
# 
# Add __Dropout__ after every pooling layer (__0.25__) and between dense layers (__0.5__).

# In[14]:


def make_model():
    """
    Define your model architecture here.
    Returns `Sequential` model.
    """
    model = Sequential()
    model.add(Conv2D(16, (3,3), padding = 'same', activation = 'relu', input_shape = (32,32,3)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32, (3,3),padding = 'same', activation = 'relu'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))    
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(LeakyReLU(0.1))
    model.add(Activation('softmax'))


    ### YOUR CODE HERE
    
    return model



# In[15]:


# describe model
#s = reset_tf_session()  # clear default graph
model = make_model()
model.summary()




# # Train model

# Training of your model can take approx. 4-8 minutes per epoch.
#
# During training you should observe the decrease in reported loss on training and validation.
#
# If the loss on training is not decreasing with epochs you should revise your model definition and learning rate.

# In[19]:


INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10

#s = reset_tf_session()  # clear default graph
# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
model = make_model()  # define our model

# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))


# Training takes approximately **1.5 hours**. You're aiming for ~0.80 validation accuracy.

# In[20]:


# we will save model checkpoints to continue training in case of kernel death
model_filename = 'cifar.{0:03d}.hdf5'
last_finished_epoch = None

#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 7
# model = load_model(model_filename.format(last_finished_epoch))


# In[ ]:

import keras_utils
# fit model
model.fit(
    x_train2, y_train2,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler),
               LrHistory(),
               keras_utils.TqdmProgressCallback(),
               keras_utils.ModelSaveCallback(model_filename)],
    validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=0,
    initial_epoch=last_finished_epoch or 0
)

# In[ ]:


# save weights to file
model.save_weights("weights.h5")

# In[ ]:


# load weights from file (can call without model.fit)
model.load_weights("weights.h5")

# # Evaluate model

# In[ ]:


# make test predictions
y_pred_test = model.predict_proba(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

# In[ ]:


# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score

plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))


# In[ ]:


# inspect preditions
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label = cifar10_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        true_label = cifar10_classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
            pred_label, pred_proba, true_label
        ))
plt.show()

# # Visualize maximum stimuli

# We want to find input images that provide maximum activations for particular layers of our network.
#
# We will find those maximum stimuli via gradient ascent in image space.
#
# For that task we load our model weights, calculate the layer output gradient with respect to image input and shift input image in that direction.

# In[ ]:


#s = reset_tf_session()  # clear default graph
K.set_learning_phase(0)  # disable dropout
model = make_model()
model.load_weights("weights.h5")  # that were saved after model.fit

# In[ ]:


# all weights we have
model.summary()


# In[ ]:


def find_maximum_stimuli(layer_name, is_conv, filter_index, model, iterations=20, step=1., verbose=True):
    def image_values_to_rgb(x):
        # normalize x: center on 0 (np.mean(x_train2)), ensure std is 0.25 (np.std(x_train2))
        # so that it looks like a normalized image input for our network
        x = (x - np.mean(x))/(np.sqrt(np.std(x))*0.25)### YOUR CODE HERE


        # do reverse normalization to RGB values: x = (x_norm + 0.5) * 255
        x =  (x + 0.5)*255 ### YOUR CODE HERE

        # clip values to [0, 255] and convert to bytes
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # this is the placeholder for the input image
    input_img = model.input
    img_width, img_height = input_img.shape.as_list()[1:3]

    # find the layer output by name
    layer_output = list(filter(lambda x: x.name == layer_name, model.layers))[0].output

    # we build a loss function that maximizes the activation
    # of the filter_index filter of the layer considered
    if is_conv:
        # mean over feature map values for convolutional layer
        loss = K.mean(layer_output[:, :, :, filter_index])
    else:
        loss = K.mean(layer_output[:, filter_index])

    # we compute the gradient of the loss wrt input image
    grads = K.gradients(loss, input_img)[0]  # [0] because of the batch dimension!

    # normalization trick: we normalize the gradient
    grads = grads / (K.sqrt(K.sum(K.square(grads))) + 1e-10)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * (0.1 if is_conv else 0.001)

    # we run gradient ascent
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if verbose:
            print('Current loss value:', loss_value)

    # decode the resulting input image
    img = image_values_to_rgb(input_img_data[0])

    return img, loss_value


# In[ ]:


# sample maximum stimuli
def plot_filters_stimuli(layer_name, is_conv, model, iterations=20, step=1., verbose=False):
    cols = 8
    rows = 2
    filter_index = 0
    max_filter_index = list(filter(lambda x: x.name == layer_name, model.layers))[0].output.shape.as_list()[-1] - 1
    fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            if filter_index <= max_filter_index:
                ax = fig.add_subplot(rows, cols, i * rows + j + 1)
                ax.grid('off')
                ax.axis('off')
                loss = -1e20
                while loss < 0 and filter_index <= max_filter_index:
                    stimuli, loss = find_maximum_stimuli(layer_name, is_conv, filter_index, model,
                                                         iterations, step, verbose=verbose)
                    filter_index += 1
                if loss > 0:
                    ax.imshow(stimuli)
                    ax.set_title("Filter #{}".format(filter_index))
    plt.show()


# In[ ]:


# maximum stimuli for convolutional neurons
conv_activation_layers = []
for layer in model.layers:
    if isinstance(layer, LeakyReLU):
        prev_layer = layer.inbound_nodes[0].inbound_layers[0]
        if isinstance(prev_layer, Conv2D):
            conv_activation_layers.append(layer)

for layer in conv_activation_layers:
    print(layer.name)
    plot_filters_stimuli(layer_name=layer.name, is_conv=True, model=model)

# In[ ]:


# maximum stimuli for last dense layer
last_dense_layer = list(filter(lambda x: isinstance(x, Dense), model.layers))[-1]
plot_filters_stimuli(layer_name=last_dense_layer.name, is_conv=False,
                     iterations=200, step=0.1, model=model)


# In[ ]:


def maximum_stimuli_test_for_grader():
    layer = list(filter(lambda x: isinstance(x, Dense), model.layers))[-1]
    output_index = 7
    stimuli, loss = find_maximum_stimuli(
        layer_name=layer.name,
        is_conv=False,
        filter_index=output_index,
        model=model,
        verbose=False
    )
    return model.predict_proba(stimuli[np.newaxis, :])[0, output_index]



# That's it! Congratulations!
#
# What you've done:
# - defined CNN architecture
# - trained your model
# - evaluated your model
# - visualised learnt filters
