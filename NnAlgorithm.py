import numpy as np
import random

"""
This file contains the core of the Neural Network Algorithm. Its constructed following the multilayer architecture

* Still in construction
"""


def sigmoid(z):
    """
    Activation function, it would be ReLu or any other activation function
    """
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig


def sigmoid_prime(z):
    sig_prime = sigmoid(z) * (1 - sigmoid(z))
    return sig_prime


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y + 1, x + 1) for x, y in zip(sizes[:-2], sizes[1:])]
        self.weights.append(np.random.randn(sizes[-1], sizes[-2]+1))

    def feedforward(self, a):
        for w in self.weights:
            a = sigmoid(np.dot(w, a))
        return a

    def gradient_descent(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(train_data)
        n = len(training_data)
        n_test = 0

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
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
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for w in self.weights:
            z = np.dot(w, activation)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations-y
