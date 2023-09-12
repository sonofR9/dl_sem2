import numpy as np
from random import Random


class NeuralLayer:
    def __init__(self):
        pass


class MLPRegressor:

    class NeuronWithWeights:
        '''
        self.output holds last output of neuron
        '''

        def __init__(self, input_num: int):
            self.bias = np.random.rand(1)
            self.weights = np.random.rand(input_num)
            self.last_output = None
            self.last_input = None
            self.activation = lambda x: x if x > 0 else 0
            self.activation_derivative = lambda x: 1 if x > 0 else 1

        def forward(self, prev_layer_out: np.ndarray) -> float:
            self.last_input = np.dot(
                prev_layer_out, self.weights) + self.bias
            self.last_output = self.activation(self.last_input)
            return self.last_output

        def backward(self, loss: float) -> np.ndarray:
            return self.weights * self.activation_derivative(self.last_input) * loss

        def update_weights(self, loss: float, prev_layer_output: np.ndarray) -> None:
            self.weights -= prev_layer_output * \
                self.activation_derivative(self.last_input) * loss

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
        for prev_size, curr_size in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self._weights += np.ndarray((prev_size, curr_size))
            self._biases += np.ndarray(curr_size)

        self._activation = lambda x: x if x > 0 else 0
        self._activation_derivative = lambda x: 1 if x > 0 else 1

    def train(self, x: np.ndarray, y: np.ndarray):
        if (self._input_size is None):
            assert self._input_size == x.shape[-1]
            assert self._output_size == y.shape[-1]
        else:
            self._init_knowing_sizes(x, y)

    def predict(self, x: np.ndarray):
        assert self._input_size is not None

    def _forward(self, single_input: np.ndarray, train: bool = False) -> None:
        assert single_input.ndim == 1

        if (train):
            self._neurons_inputs = []
            # save input to neural network (as output of input layer)
            self._neurons_outputs = [single_input]

        last_layer_out = single_input
        for weights, biases in zip(self._weights, self._biases):
            neurons_input = last_layer_out @ weights + biases
            last_layer_out = self._activation(neurons_input)
            if (train):
                self._neurons_inputs += neurons_input
                self._neurons_outputs += last_layer_out

        return last_layer_out

    def _update_weights(self, loss: np.ndarray) -> None:
        dl_dout = -2*loss
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
        self._input_size = x.shape[1]
        first_layer_size = self._weights[0].shape[0]
        self._weights.insert(0, np.ndarray(
            (self._input_size, first_layer_size)))
        self._biases.insert(0, np.ndarray(first_layer_size))

        self._output_size = y.shape[-1]
        last_layer_size = self._weights[-1].shape[1]
        self._weights += np.ndarray((last_layer_size, self._output_size))
        self._biases += np.ndarray(last_layer_size)
