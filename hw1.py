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

    def train(self, X: np.ndarray, Y: np.ndarray):
        if (self._input_size):
            assert (self._input_size == X.shape[-1])
            assert (self._output_size == Y.shape[-1])
        else:
            self._input_size = X.shape[1]
            first_layer_size = self._weights[0].shape[0]
            self._weights.insert(0, np.ndarray(
                (self._input_size, first_layer_size)))
            self._biases.insert(0, np.ndarray(first_layer_size))

            self._output_size = Y.shape[-1]
            last_layer_size = self._weights[-1].shape[1]
            self._weights += np.ndarray(
                (last_layer_size, self._output_size))
            self._biases += np.ndarray(last_layer_size)

        pass

    def predict(self, X):
        pass
