import matplotlib.pyplot as plt
import numpy as np


class Activation:
    def forward(self, inp: np.ndarray) -> np.ndarray:
        return inp

    def backward(self, inp: np.ndarray) -> np.ndarray:
        return 1


class Linear(Activation):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        return inp

    def backward(self, inp: np.ndarray) -> np.ndarray:
        return 1


class LeakedReLu(Activation):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))


class Layer:
    def __init__(
        self, prev_size: int, size: int, learning_rate: float, activation: Activation
    ):
        self.learning_rate = learning_rate
        self.biases = np.random.rand(size).reshape(-1, 1) / 10
        if prev_size is not None and prev_size != 0:
            self.weights = np.random.rand(size, prev_size) / 10

        self._activation = activation

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 0:
            self.neurons_input = self.weights * input + self.biases
        else:
            self.neurons_input = self.weights @ input + self.biases
        return self._activation.forward(self.neurons_input)

    def backward_with_update(
        self, error: np.ndarray, prev_out: np.ndarray
    ) -> np.ndarray:
        dl_dout = error.reshape(-1, 1)
        limit = 1
        dl_dout = np.where(dl_dout < limit, dl_dout, limit)
        dl_dout = np.where(dl_dout > -limit, dl_dout, -limit)

        neurons_backward = dl_dout * self._activation.backward(self.neurons_input)
        neurons_backward = neurons_backward.reshape(-1)

        dl_dw = np.outer(neurons_backward, prev_out)
        dl_db = neurons_backward

        dl_dout = self.weights.T @ neurons_backward

        self.weights -= self.learning_rate * dl_dw
        self.biases -= self.learning_rate * dl_db.reshape(-1, 1)

        return dl_dout


class MLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        learning_rate: float = 0.001,
        max_iter: int = 10,
        activation: Activation = Sigmoid(),
    ):
        self._learning_rate = learning_rate
        self._max_iter = max_iter

        self._layers = []

        self._input_size = None
        self._output_size = None

        self._layers.append(
            Layer(None, hidden_layer_sizes[0], learning_rate, activation)
        )
        for prev_size, curr_size in zip(hidden_layer_sizes, hidden_layer_sizes[1:]):
            self._layers.append(Layer(prev_size, curr_size, learning_rate, activation))

        self._loss = lambda actual, predicted: (actual - predicted) ** 2
        self._loss_derivative = lambda actual, predicted: -2 * (actual - predicted)

    def train(self, x: np.ndarray, y: np.ndarray):
        if self._input_size is not None:
            if x.ndim == 1:
                assert (
                    self._input_size == 1
                ), "Input size differs from its previous value!"
            else:
                assert (
                    self._input_size == x.shape[-1]
                ), "Input size differs from its previous value!"
            if y.ndim == 1:
                assert (
                    self._output_size == 1
                ), "output size differs from its previous value!"
            else:
                assert (
                    self._output_size == y.shape[-1]
                ), "output size differs from its previous value!"
        else:
            self._init_knowing_sizes(x, y)

        losses = []
        for epoch in range(self._max_iter):
            for inp, actual in zip(x, y):
                predicted = self._forward(inp, True)
                loss = self._loss(actual, predicted)
                self._update_weights(actual, predicted)
                losses.append(loss)
        plt.plot(np.array(losses)[:, 0])
        plt.plot(np.array(losses)[:, 1])
        plt.show()

    def predict(self, x: np.ndarray):
        assert self._input_size is not None, "Neural network must be trained first!"
        if x.ndim == 1:
            assert (
                self._input_size == 1
            ), """Input size differs from training! 
                    current = 1, during training = {self._input_size}"""
        else:
            assert (
                self._input_size == x.shape[-1]
            ), """Input size differs from training! 
                    current = {x.shape[-1]}, during training = {self._input_size}"""

        result = []
        if x.ndim == 2 or (x.ndim == 1 and self._input_size == 1):
            for inp in x:
                result.append(self._forward(inp))
        else:
            result = self._forward(x)

        return np.array(result)

    def _forward(self, single_input: np.ndarray, train: bool = False) -> np.ndarray:
        assert single_input.ndim == 1 or single_input.ndim == 0

        if train:
            self._neurons_outputs = [single_input]

        last_layer_out = single_input.reshape(-1, 1)

        for layer in self._layers:
            last_layer_out = layer.forward(last_layer_out)
            if train:
                self._neurons_outputs.append(last_layer_out)

        return last_layer_out.reshape(-1)

    def _update_weights(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        dl_dout = self._loss_derivative(actual, predicted)

        for i in range(len(self._layers) - 1, -1, -1):
            dl_dout = self._layers[i].backward_with_update(
                dl_dout, self._neurons_outputs[i]
            )

    def _init_knowing_sizes(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim == 1:
            self._input_size = 1
        else:
            self._input_size = x.shape[1]
        first_layer_size = self._layers[0].biases.size
        self._layers[0].weights = (
            np.random.rand(first_layer_size, self._input_size) / 10
        )

        if y.ndim == 1:
            self._output_size = 1
        else:
            self._output_size = y.shape[1]
        last_layer_size = self._layers[-1].biases.size
        self._layers.append(
            Layer(last_layer_size, self._output_size, self._learning_rate, Linear())
        )


# Generate a dataset
x = np.linspace((0, 0), (1000, 1000), 100)
y = x * x + 2 * x + 1

# Create an MLPRegressor object
regressor = MLPRegressor(
    hidden_layer_sizes=(30,),
    learning_rate=0.001,
    max_iter=100,
    activation=LeakedReLu(1),
)

# Train the regressor
regressor.train(x, y)

# Predict the values of y for the given values of x
predicted_y = regressor.predict(x)

# Plot the results
plt.figure()
# plt.plot(x, y, label="Actual")
# plt.plot(x, predicted_y.reshape(-1), label="Predicted")
plt.legend()
plt.show()
input()
