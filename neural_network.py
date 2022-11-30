import numpy as np
from nnfs.datasets import spiral_data, vertical_data # dataset examples

from lib.layer_dense import LayerDense
from lib.loss import Loss, CategoricalCrossEntropy
from lib.activation_functions import Activation_ReLU, Activation_SoftMax


def spiral_data_network(X, y):
    activation1 = Activation_ReLU()
    activation2 = Activation_SoftMax()

    dense_1 = LayerDense(2,3)
    dense_2 = LayerDense(3,3)

    loss_function = CategoricalCrossEntropy()

    dense_1.forward(X)
    activation1.forward(dense_1.output)

    dense_2.forward(activation1.output)
    activation2.forward(dense_2.output)

    print(activation2.output[:5])


    loss = loss_function.calculate(activation2.output, y)
    print(f'Loss: {loss}')

    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions==y)
    print(f'Accuracy: {accuracy}')


def vertical_data_network(X, y):
    activation1 = Activation_ReLU()
    activation2 = Activation_SoftMax()

    dense_1 = LayerDense(2,3)
    dense_2 = LayerDense(3,3)

    loss_function = CategoricalCrossEntropy()

    lowest_loss = 9999999

    best_dense_1_weights = dense_1.weights.copy()
    best_dense_1_biases = dense_1.biases.copy()

    best_dense_2_weights = dense_2.weights.copy()
    best_dense_2_biases = dense_2.biases.copy()

    for i in range(10000):
        dense_1.weights = np.random.randn(2,3) * 0.05
        dense_2.biases = np.random.randn(1,3) * 0.05

        dense_2.weights = np.random.randn(3,3) * 0.05
        dense_2.biases = np.random.randn(1,3)* 0.05

        dense_1.forward(X)
        activation1.forward(dense_1.output)

        dense_2.forward(activation1.output)
        activation2.forward(dense_2.output)

        loss = loss_function.calculate(activation2.output, y)

        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions==y)

        if loss < lowest_loss:
            print('New set of weights found, iteration: ', i, 'loss: ', loss, 'acc: ', accuracy )

            best_dense_1_weights = dense_1.weights.copy()
            best_dense_1_biases = dense_1.biases.copy()

            best_dense_2_weights = dense_2.weights.copy()
            best_dense_1_biases = dense_2.biases.copy()

            lowest_loss = loss

X_spiral, y_spiral = spiral_data(samples=100, classes=3)
spiral_data_network(X_spiral, y_spiral)

X_vertical, y_vertical = vertical_data(samples=100, classes=3)
vertical_data_network(X_vertical, y_vertical)
