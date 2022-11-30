import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.biases = np.zeros((1,n_neurons)) # setting default bias values to zero.
        self.weights = np.random.randn(n_inputs, n_neurons) # setting random values of weights.


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # calculating the output of each neuron.


