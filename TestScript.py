import mnist_loader
import NN_core

"""
In this file, the learning algorithm is trained with the MNIST dataset, the examples are
loaded using the structure raised by Michael Nielsen, the input array has 50.000 examples
each one with 784 inputs and 10 outputs 
"""


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = NN_core.Network([784, 30, 30, 10])
net.sgd(training_data, 10, 10, 3.0, test_data=test_data)
