import numpy as np
import random

"""
This file contains the core of the Neural Network Algorithm. Its constructed following the multilayer architecture

* Still in construction
"""


# TODO Add a cross validation feature
# TODO Add different activation functions
# TODO Add different cost functions,
#  now the algorithm implement the squared difference, in the future it will implement the cost for logistic
#  regression adding regularization, etc


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
        # TODO Make all computations as matrix operations
        # Sizes, list with the number of neurons of each layer
        # The number of layers is defined by the size of the "sizes" list
        self.num_layers = len(sizes)
        # Like in the Harvard course, the activations were modified to add the
        # column of ones at the beginning of the array.
        self.sizes = [sizes[x] + 1 if x < len(sizes) - 1 else sizes[x]
                      for x in range(len(sizes))]
        # The weight values were initialized
        # On Machine Learning course by Harvard the weights were named "theta"
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # Matrix function
        # feedforward propagation
        for w in self.weights:
            a = sigmoid(np.dot(w, a))
        return a

    def gradient_descent(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        # TODO Implement the gradient descent as matrix operation,
        #  only conserve the for loop to the epochs iteration
        training_data = list(train_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        else:
            n_test = 0

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {}% - {}/{}".format(j,
                                                      self.evaluate(test_data) * 100 / n_test,
                                                      self.evaluate(test_data),
                                                      n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        # TODO Update the weights matrix deleting the for loop
        delta_w = [np.zeros(w.shape) for w in self.weights]
        x_train, y_train = zip(*mini_batch)
        x_train = np.array(x_train)
        x_train = x_train.reshape(-1, len(x_train[-1]))
        y_train = np.array(y_train)
        nabla_w = self.backpropagation(x_train, y_train)
        delta_w = [nw + dnw for nw, dnw in zip(delta_w, nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, delta_w)]

    def backpropagation(self, x, y):
        # TODO Delete the for loops, implement matrix operations!!!
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        print(len(activation[0]))
        activations = [activation]
        zs = []

        for w in self.weights:
            z = []
            for i in range(len(activation[1])):
                z.append(np.dot(w.transpose(), activation[:,i]))
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return delta_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations - y
