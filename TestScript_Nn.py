import mnist_loader
import Nn
import numpy as np

"""
In this file, the learning algorithm is trained with the MNIST dataset, the examples are
loaded using the structure raised by Michael Nielsen, the input array has 50.000 examples
each one with 784 inputs and 10 outputs 
"""


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Unpacking the training_data and the test_data to modify them adding the ones
# column at the beginning of each array.w
# x, y = zip(*training_data)
# x_test, y_test = zip(*test_data)
# x_test = np.insert(x_test, 0, 1.0, axis=1)
# x = np.insert(x, 0, 1.0, axis=1)
# x = x[:1]
# y = y[:1]


# Now the training and test data is packed using a zip object to use as input of the NnAlgorithm module
# training_data = zip(x, y)
# test_data = zip(x_test, y_test)

# Network takes as input a list
net = Nn.Network([784, 30, 30, 10])

# gradient_descent takes as inputs (training_data, epochs, mini_batch_size, eta, test_data(optional))
net.gradient_descent(training_data, 1, 10, 3.0, test_data=test_data)
