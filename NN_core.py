import numpy as np
import random

# The activation function and its derivative are defined.


def sigmoid(z):
    sig = 1.0/(1.0+np.exp(-z))
    return sig


def sigmoid_prime(z):
    sig_prime = sigmoid(z)*(1-sigmoid(z))
    return sig_prime


class Network(object):
    def __init__(self, sizes):
        # Sizes is a list with the neuron's number of each layer
        # The number of layers its equal to len(sizes)
        self.num_layers = len(sizes)
        self.sizes = sizes
        # The bias are randomly initialized
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # The weights are randomly initialized
        # On Harvard course the weights are named as theta
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Forward propagation, the activation is calculated with the sigmoid function
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def sgd(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        # Stochastic gradient descent algorithm
        training_data = list(train_data)
        n = len(training_data)
        n_test = 0

        # If test_data is not None, convert it to a list and calculate n_test as len(test_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            # The training data are shuffled
            random.shuffle(training_data)
            # Creating mini batches with the training data
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {}% - {}/{}".format(j, self.evaluate(test_data)*100/n_test, self.evaluate(test_data),
                                                      n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # nabla bias and nabla weights matrices are initialized with zero values
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # The input is the initial array x
        activation = x
        # "Activations" contains the activation of each layer, it's a list of lists.
        activations = [x]
        # "zs" is a list to save the "z" value of each layer
        zs = []

        # Computing the forward propagation
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        return output_activations-y
