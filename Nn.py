import numpy as np
import random

"""
This file contains the core of the Neural Network Algorithm.
"""


def sigmoid(z):
    """
    Activation function. Implement matrix operations
    """
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig


def sigmoid_prime(z):
    """
    Derivative of the activation function. Implement matrix operations
    """
    sig_prime = np.multiply(sigmoid(z), (1 - sigmoid(-z)))
    return sig_prime


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = [sizes[x] + 1 if x < len(sizes) - 1 else sizes[x]
                      for x in range(len(sizes))]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # This function must work with a loop because the activation used for each layer depends on the
        for w in self.weights:
            a = sigmoid(np.dot(w, a))
        return a

    def gradient_descent(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        # The argument train_data is a zip object made up of two lists, one list is the initial activation or the
        # entries of the network and the other list is the classification for this activation array that is to say
        # the outputs of the network **(supervised learning).

        # While converting the zip object into a list, each element of the list is a tuple (example) made up of
        # input and output
        training_data = list(train_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        else:
            n_test = 0

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {}% - {}/{}".format(j, self.evaluate(test_data)*100/n_test, self.evaluate(test_data),
                                                      n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        print(self.weights[0].shape)
        print(nabla_w[0].shape)
        self.weights = [w - (eta * (nw/len(mini_batch))) for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_l = []
        activation = x
        activations = [x]

        zs = []

        for w in self.weights:
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # delta is the error of the computation at the end of the network,
        # error between output_calculated and output_desired
        delta = self.error_derivative(activations[-1], y)
        print(delta.shape)
        delta_l.append(delta)

        # Calculation of the delta's of each layer, naturally the delta_l matrix has a dimension of (num_layers - 1)
        for layer in range(2, self.num_layers):
            # Its used the Hadamard multiplication or element wise multiplication noted by np.multiply(a,b)
            print(delta_l[layer - 2].shape)
            delta_layer = np.multiply(np.inner(self.weights[-layer+1].transpose(), delta_l[layer-2]).transpose(),
                                      sigmoid_prime(zs[-layer]))

            delta_l.append(delta_layer)

        # delta_l.reverse()
        delta_l = delta_l[::-1]
        for i in range(len(self.weights)):
            nabla_w[i] = delta_l[i] * activations[i].transpose()
        # nabla_w = np.dot(delta_l, activations[1:].transpose())
        return nabla_w

    # In this case is a static method because the function only does a simple calculation between the arguments,
    # this function was extracted from the backpropagation method because it could be use any other algorithm
    # to calculate the error, here is used the quadratic error function (1/2)*(calculated_output - desired_output)**2,
    # and the error derivative is (calculated_output - desired_output)
    @staticmethod
    def error_derivative(calculated_output, desired_output):
        # calculated_output is an array of the outputs calculated by the network
        # desired_output is an array of the outputs that we desire
        return calculated_output - desired_output

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
