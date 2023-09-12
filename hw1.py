import matplotlib as plt
import numpy as np
from random import Random


class MLPRegressor:

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
        for prev_size, curr_size in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self._weights += [np.random.rand(prev_size, curr_size)]
            self._biases += [np.random.rand(curr_size)]

        self._activation = lambda x: x if x > 0 else 0
        self._activation_derivative = lambda x: 1 if x > 0 else 1

        self._loss = lambda correct, actual: (correct-actual)**2
        self._loss_derivative = lambda loss: -2*loss

    def train(self, x: np.ndarray, y: np.ndarray):
        if (self._input_size is not None):
            if (x.ndim == 1):
                assert self._input_size == 1, "Input size differs from its previous value!"
            else:
                assert self._input_size == x.shape[-1], "Input size differs from its previous value!"
            assert self._output_size == y.shape[-1], "output size differs from its previous value!"
        else:
            self._init_knowing_sizes(x, y)

        for epoch in range(self._max_iter):
            for inp, correct in zip(x, y):
                actual = self._forward(inp, True)
                loss = self._loss(correct, actual)
                self._update_weights(loss)

    def predict(self, x: np.ndarray):
        assert self._input_size is not None, "Neural network must be trained first!"
        if (x.ndim == 1):
            assert self._input_size == 1, """Input size differs from training! 
                    current = 1, during training = {self._input_size}"""
        else:
            assert self._input_size == x.shape[-1], """Input size differs from training! 
                    current = {x.shape[-1]}, during training = {self._input_size}"""

        result = []
        if (x.ndim == 2):
            for inp in x:
                result += self._forward(inp)
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

        if (single_input.ndim == 0):
            for weights, biases in zip(self._weights, self._biases):
                neurons_input = last_layer_out * weights + biases

                print(neurons_input.shape)

                last_layer_out = np.apply_along_axis(
                    self._activation, 0, neurons_input)
                if (train):
                    self._neurons_inputs += [neurons_input]
                    self._neurons_outputs += [last_layer_out]

            return last_layer_out

        for weights, biases in zip(self._weights, self._biases):

            neurons_input = last_layer_out @ weights + biases
            last_layer_out = np.apply_along_axis(
                self._activation, 0, neurons_input)
            if (train):
                self._neurons_inputs += [neurons_input]
                self._neurons_outputs += [last_layer_out]

        return last_layer_out

    def _update_weights(self, loss: np.ndarray) -> None:
        dl_dout = self._loss_derivative(loss)
        for i in range(len(self._weights)-1, -1, -1):
            neurons_backward = self._activation_derivative(
                self._neurons_inputs[i])
            dout_dw = np.outer(neurons_backward, self._neurons_outputs[i])
            dl_dw = dout_dw @ dl_dout

            dout_db = neurons_backward @ self._biases[i].T
            dl_db = dout_db @ dl_dout

            dcurr_dprev = neurons_backward @ self._weights[i].T
            dl_dout = dcurr_dprev @ dl_dout

            self._weights[i] -= self._learning_rate * dl_dw
            self._biases -= self._learning_rate * dl_db

    def _init_knowing_sizes(self, x: np.ndarray, y: np.ndarray) -> None:
        if (x.ndim == 1):
            self._input_size = x.shape[0]
        else:
            self._input_size = x.shape[1]
        first_layer_size = self._weights[0].shape[0]
        self._weights.insert(0, np.random.rand(
            self._input_size, first_layer_size))
        self._biases.insert(0, np.random.rand(first_layer_size))

        self._output_size = y.shape[-1]
        last_layer_size = self._weights[-1].shape[1]
        self._weights += [np.random.rand(last_layer_size, self._output_size)]
        self._biases += [np.random.rand(last_layer_size)]


# Generate a dataset
x = np.linspace(0, 1, 100)
y = x * x + 2 * x + 1

# Create an MLPRegressor object
regressor = MLPRegressor(hidden_layer_sizes=(
    100, 10), learning_rate=0.001, max_iter=10)

# Train the regressor
regressor.train(x, y)

# Predict the values of y for the given values of x
predicted_y = regressor.predict(x)

# Plot the results
plt.plot(x, y, label="Actual")
plt.plot(x, predicted_y, label="Predicted")
plt.legend()
plt.show()
