import numpy as np


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # using ReLU activation function on the input.


class Activation_SoftMax:
    def forward(self, inputs):
        # applying softmax activation function on the input
        exp_values = np.exp(inputs - np.max(inputs, keepdims=True, axis=1))
        probabilities = exp_values / np.sum(exp_values, keepdims=True, axis=1)

        self.output = probabilities
