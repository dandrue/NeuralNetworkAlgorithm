# NeuralNetworkAlgorithm
Custom Neural Network Algorithm

This repository contains a custom Neural Network Algorithm based on the code proposed by Michael Nielsen in "Neural Networks and Deep Learning" (http://neuralnetworksanddeeplearning.com/) and the course "Machine Learning" offered by Standford and instructed by Andrew NG, this course is available in Coursera.

The principal contribution and main difference between the code in this repository and the code by the references are that I implemented matrix computations to work with the weights and biases of the network. You could depict that the bias treatment differs from code to code, here, to improve the calculations, I decided to include it into the input array giving an extra abstraction to the calculations. For a most loyal representation to theory, you could refer to the Michael Nielsen book cited above.

The MNIST dataset is used to test the algorithm. Initially, the gradient descent algorithm is implemented to train the network, future perspectives will implement different algorithms to train the network, like genetic algorithms and Particle Swarm Optimization algorithms.
