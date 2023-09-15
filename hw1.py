import matplotlib.pyplot as plt
import numpy as np
from random import Random


class MLPRegressor:

    class Layer:
        def __init__(self):
            pass

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        learning_rate: float = 0.001,
        max_iter: int = 10,
    ):
        self._learning_rate = learning_rate
        self._max_iter = max_iter

        # everything should be as close to numpy arrays as possible for speed therefore no Neuron class
        self._input_size = None
        self._output_size = None
        self._weights = []
        self._biases = []

        # TODO consider one hidden layer
        self._biases.append(np.random.rand(hidden_layer_sizes[0]))
        for prev_size, curr_size in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self._weights.append(np.random.rand(curr_size, prev_size))
            self._biases.append(np.random.rand(curr_size))

        self._activation = lambda x: np.where(x > 0, x, 0)
        self._activation_derivative = lambda x: np.where(x > 0, 1, 0)

        self._loss = lambda actual, predicted: (actual-predicted)**2
        self._loss_derivative = lambda actual, predicted: -2*(actual-predicted)

    def train(self, x: np.ndarray, y: np.ndarray):
        if (self._input_size is not None):
            if (x.ndim == 1):
                assert self._input_size == 1, "Input size differs from its previous value!"
            else:
                assert self._input_size == x.shape[-1], "Input size differs from its previous value!"
            if (y.ndim == 1):
                assert self._output_size == 1, "output size differs from its previous value!"
            else:
                assert self._output_size == y.shape[-1], "output size differs from its previous value!"
        else:
            self._init_knowing_sizes(x, y)

        losses = []
        for epoch in range(self._max_iter):
            for inp, actual in zip(x, y):
                predicted = self._forward(inp, True)
                loss = self._loss(actual, predicted)
                self._update_weights(actual, predicted)
                losses.append(loss)
        plt.plot(losses)
        plt.show()

    def predict(self, x: np.ndarray):
        assert self._input_size is not None, "Neural network must be trained first!"
        if (x.ndim == 1):
            assert self._input_size == 1, """Input size differs from training! 
                    current = 1, during training = {self._input_size}"""
        else:
            assert self._input_size == x.shape[-1], """Input size differs from training! 
                    current = {x.shape[-1]}, during training = {self._input_size}"""

        result = []
        if (x.ndim == 2 or (x.ndim == 1 and self._input_size == 1)):
            for inp in x:
                result.append(self._forward(inp))
        else:
            result = self._forward(x)

        return result

    def _forward(self, single_input: np.ndarray, train: bool = False) -> np.ndarray:
        assert single_input.ndim == 1 or single_input.ndim == 0

        if (train):
            self._neurons_inputs = []
            # save input to neural network (as output of input layer)
            self._neurons_outputs = [single_input]

        last_layer_out = single_input

        i = 0
        for weights, biases in zip(self._weights, self._biases):
            i += 1
            if (last_layer_out.ndim == 0):
                neurons_input = weights * \
                    last_layer_out + biases.reshape(-1, 1)
            else:
                neurons_input = weights @ last_layer_out + \
                    biases.reshape(-1, 1)

            last_layer_out = self._activation(neurons_input)
            if (train):
                self._neurons_inputs.append(neurons_input)
                self._neurons_outputs.append(last_layer_out)

        return last_layer_out.reshape(-1)

    def _update_weights(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        dl_dout = self._loss_derivative(actual, predicted)
        for i in range(len(self._weights)-1, -1, -1):
            last_input = self._neurons_inputs[i]
            prev_out = self._neurons_outputs[i]
            weights = self._weights[i]
            bias = self._biases[i]

            neurons_backward = dl_dout * \
                self._activation_derivative(self._neurons_inputs[i])
            neurons_backward = neurons_backward.reshape(-1)

            dl_dw = np.outer(neurons_backward, self._neurons_outputs[i])
            dl_db = neurons_backward

            self._weights[i] -= self._learning_rate * dl_dw
            self._biases[i] -= self._learning_rate * dl_db

            dl_dout = self._weights[i].T @ neurons_backward
            dl_dout = dl_dout.reshape(-1, 1)

    def _init_knowing_sizes(self, x: np.ndarray, y: np.ndarray) -> None:
        if (x.ndim == 1):
            self._input_size = 1
        else:
            self._input_size = x.shape[1]
        first_layer_size = self._biases[0].size
        self._weights.insert(0, np.random.rand(first_layer_size,
                                               self._input_size))

        if (y.ndim == 1):
            self._output_size = 1
        else:
            self._output_size = y.shape[1]
        last_layer_size = self._biases[-1].size
        self._weights.append(np.random.rand(
            self._output_size, last_layer_size))
        self._biases.append(np.random.rand(self._output_size))


# Generate a dataset
x = np.linspace(0, 1, 100)
y = x * x + 2 * x + 1
y = np.ones_like(x)

# Create an MLPRegressor object
regressor = MLPRegressor(hidden_layer_sizes=(
    3, 2), learning_rate=0.001, max_iter=10)

# Train the regressor
regressor.train(x, y)

# Predict the values of y for the given values of x
predicted_y = regressor.predict(x)

# Plot the results
plt.plot(x, y, label="Actual")
# plt.plot(x, predicted_y[:][0], label="Predicted")
# plt.legend()
# plt.show()
input()
