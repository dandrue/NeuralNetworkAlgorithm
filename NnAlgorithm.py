import numpy as np
import pandas as pd
import random


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
        self.sizes = sizes
        # The weight values were initialized
        # On Machine Learning course by Harvard the weights were named "theta"
        self.weights = [np.random.randn(y, x+1) for x, y in zip(sizes[:-1], sizes[1:])]
        # print(self.weights)
        a = pd.DataFrame(self.weights)
        print(len(a[0][0]))

    def feedforward(self, a):
        # feedforward propagation
        for w in self.weights:
            a = sigmoid(np.dot(w, a))
        return a

    def gradient_descent(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(train_data)
        n = len(training_data)
        training_data[0] = np.ones(n)
        for i in training_data[0:1]:
            print(len(i[0]))
            print(i[0])
            print(len(i[1]))
            print(i[1])

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        else:
            n_test = 0

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # print(mini_batch)
                # break
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {}% - {}/{}".format(j,
                                                      self.evaluate(test_data) * 100 / n_test,
                                                      self.evaluate(test_data),
                                                      n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            nabla_w = self.backpropagation(x, y)
            delta_w = [nw + dnw for nw, dnw in zip(delta_w, nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, delta_w)]

    def backpropagation(self, x, y):
        delta_w = [np.zeros(w.shape) for w in self.weights]

        activation = x

        activations = [x]

        zs = []

        for w in self.weights:
            z = np.dot(w, activation)
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
