import numpy as np
import sys
import tqdm_utils
import download_utils
from IPython.display import clear_output
from util import eval_numerical_gradient
import matplotlib.pyplot as plt
from preprocessed_mnist import load_dataset


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        pass

    def forward(self, input):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        """
        # A dummy layer just returns whatever it gets as input.
        return input
    def forward_with_regularization(self, input, parameters):
        return input

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.

        To compute loss gradients w.r.t input, you need to apply chain rule (backprop):

        d loss / d x  = (d loss / d layer) * (d layer / d x)

        Luckily, you already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.

        If your layer has parameters (e.g. dense layer), you also need to update them here using d loss / d layer
        """
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        num_units = input.shape[1]

        d_layer_d_input = np.eye(num_units)
        print(d_layer_d_input)

        return np.dot(grad_output, d_layer_d_input)  # chain rule
    def backward_with_regularization(self, input, grad_output, lambd):
        return self.backward(input, grad_output)
class ReLU(Layer):
    def __init__(self):
        """ReLU layer simply applies elementwise rectified linear unit to all inputs"""
        pass

    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        reLu = np.maximum(0, input)  # <your code. Try np.maximum>
        return reLu

    def forward_with_regularization(self, input, parameters):
        return self.forward(input)

    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output * relu_grad

    def backward_with_regularization(self, input, grad_output, lambd):
        """Same as backward"""
        return self.backward(input, grad_output)
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate

        # initialize weights with small random numbers. We use normal initialization,
        # but surely there is something better. Try this once you got it working: http://bit.ly/2vTlmaJ
        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        f = np.dot(input, self.weights) + self.biases
        return f
    def forward_with_regularization(self, input, parameters):
        parameters.append(self.weights)
        return self.forward(input)

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        d_dense_d_input = self.weights.T
        grad_input = np.dot(grad_output, d_dense_d_input)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input

    def backward_with_regularization(self, input, grad_output, lambd):
        d_dense_d_input = self.weights.T
        grad_input = np.dot(grad_output, d_dense_d_input)

        grad_weights = np.dot(input.T, grad_output) + lambd*self.weights
        grad_biases = np.sum(grad_output, axis=0)
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        self.weights = self.weights - self.learning_rate*grad_weights
        self.biases = self.biases - self.learning_rate*grad_biases

        return grad_input


# test gradients w.r.t. params
def compute_out_given_wb(w, b):
    l = Dense(32, 64, learning_rate=1)
    l.weights = np.array(w)
    l.biases = np.array(b)
    x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
    return l.forward(x)
def compute_grad_by_params(w, b):
    l = Dense(32, 64, learning_rate=1)
    l.weights = np.array(w)
    l.biases = np.array(b)
    x = np.linspace(-1, 1, 10 * 32).reshape([10, 32])
    l.backward(x, np.ones([10, 64]) / 10.)
    return w - l.weights, b - l.biases
def softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy
def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)
plt.figure(figsize=[6,6])
for i in range(3):
    plt.subplot(2,2,i+1)
    plt.title("Label: %i"%y_train[i])
    plt.imshow(X_train[i].reshape([28,28]),cmap='gray')



network = []
network.append(Dense(X_train.shape[1],100))
network.append(ReLU())
network.append(Dense(100,200))
network.append(ReLU())
network.append(Dense(200,100))
network.append(ReLU())
network.append(Dense(100,10))


def forward(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer.
    Make sure last activation corresponds to network logits.
    """
    activations = []
    input = X

    # <your code here>
    for act in network:
        tmp = act.forward(input)
        activations.append(tmp)
        input = tmp



    assert len(activations) == len(network)
    return activations


def predict(network, X):
    """
    Compute network predictions.
    """
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    """
    Train your network on a given batch of X and y.
    You first need to run forward to get all layer activations.
    Then you can run layer.backward going from last to first layer.

    After you called backward for all layers, all Dense layers have already made one gradient step.
    """

    # Get the layer activations
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    # <your code: propagate gradients through the network>
    for i in range(len(network) - 1, -1, -1):
        loss_grad = network[i].backward(layer_inputs[i], loss_grad)

    return np.mean(loss)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in tqdm_utils.tqdm_notebook_failsafe(range(0, len(inputs) - batchsize + 1, batchsize)):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


train_log = []
val_log = []
def forward_with_regularization(network, X, parameters):
    activations = []
    input = X
    for net in network:
        output = net.forward_with_regularization(input, parameters)
        activations.append(output)
        input = output
    assert len(activations) == len(network)
    return activations

def train_with_regularization(network, X,y, parameters, lamb):
    layer_activations = forward_with_regularization(network, X, parameters)
    layer_inputs = [X] + layer_activations
    logits = layer_activations[-1]

    #Computing loss and initial gradient
    cross_entropy_cost = np.mean(softmax_crossentropy_with_logits(logits,y))
    L2_regularization_cost = np.sum(np.sum(W) for W in parameters)*lamb/(2*y.shape[0])
    loss = cross_entropy_cost + L2_regularization_cost
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    #Propagating gradients through the network
    for i in range(len(network)-1, -1, -1):
        loss_grad = network[i].backward_with_regularization(layer_inputs[i], loss_grad , lamb)

    return loss


for epoch in range(25):
    for x_batch, y_batch in iterate_minibatches(X_train, y_train, batchsize=32, shuffle=True):
        parameters = []
        train_with_regularization(network, x_batch, y_batch, parameters = [], lamb=0.0025)

    train_log.append(np.mean(predict(network, X_train) == y_train))
    val_log.append(np.mean(predict(network, X_val) == y_val))

    clear_output()
    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Val accuracy:", val_log[-1])
    plt.plot(train_log, label='train accuracy')
    plt.plot(val_log, label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


