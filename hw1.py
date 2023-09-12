import numpy as np
from random import Random


class NeuralLayer:
    def __init__(self):
        pass


class MLPRegressor:

    class NeuronWeight:
        '''
        self.output holds last output of neuron
        '''

        def __init__(self, input_num: int):
            self.bias = np.random.rand(1)
            self.weights = np.random.rand(input_num)
            self.output = None
            self.activation = lambda x: x if x > 0 else 0
            self.activation_derivative = lambda x: 1 if x > 0 else 1

        def forward(self, prev_layer_output: np.ndarray) -> float:
            self.last_input = np.dot(
                prev_layer_output, self.weights) + self.bias
            self.last_output
            return self.output

        def last_output(self) -> float:
            return self.output

        def update_weights(self, loss: float, prev_layer_output: np.ndarray) -> None:
            self.weights -= prev_layer_output  # * activation_function(loss)

        def backward(self, loss: float) -> np.ndarray:
            return 1

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        learning_rate=0.001,
        max_iter=10,
    ):
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._layers = hidden_layer_sizes
        # self.

    def train(self, X, y):
        pass

    def predict(self, X):
        pass
