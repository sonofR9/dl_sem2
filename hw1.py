import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


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
        self.dl_db = np.zeros_like(self.biases)
        if prev_size is not None and prev_size != 0:
            self.weights = np.random.rand(size, prev_size) / 10
            self.dl_dw = np.zeros_like(self.weights)

        self._activation = activation

        self._backward_count = 0

    def forward(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 0:
            self.neurons_input = self.weights * input + self.biases
        else:
            self.neurons_input = self.weights @ input + self.biases
        return self._activation.forward(self.neurons_input)

    def backward(self, error: np.ndarray, prev_out: np.ndarray) -> np.ndarray:
        self._backward_count += 1

        dl_dout = error.reshape(-1, 1)
        # limit = 1
        # dl_dout = np.where(dl_dout < limit, dl_dout, limit)
        # dl_dout = np.where(dl_dout > -limit, dl_dout, -limit)

        neurons_backward = dl_dout * self._activation.backward(self.neurons_input)
        neurons_backward = neurons_backward.reshape(-1)

        self.dl_dw += np.outer(neurons_backward, prev_out)
        self.dl_db += neurons_backward.reshape(-1, 1)

        dl_dout = self.weights.T @ neurons_backward
        return dl_dout

    def update_weights(self) -> None:
        self.weights -= self.learning_rate * self.dl_dw / self._backward_count
        self.biases -= self.learning_rate * self.dl_db / self._backward_count

        self.dl_dw = np.zeros_like(self.dl_dw)
        self.dl_db = np.zeros_like(self.dl_db)
        self._backward_count = 0


class MLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        learning_rate: float = 0.001,
        max_iter: int = 10,
        activation: Activation = Sigmoid(),
        batch: int = 10,
    ):
        self._learning_rate = learning_rate
        self._max_iter = max_iter
        self._batch = batch

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

        self._x_norm_coef = x.mean()
        x = x / self._x_norm_coef

        self._y_norm_coef = y.mean()
        y = y / self._y_norm_coef

        losses = []

        for _ in range(self._max_iter):
            i = 0
            losses_buf = []
            for inp, actual in zip(x, y):
                predicted = self._forward(inp, True)
                loss = self._loss(actual, predicted)
                self._backward(actual, predicted)
                i += 1
                losses_buf.append(loss * self._y_norm_coef)
                if i % self._batch == 0:
                    self._update_weights()
                if i % int(x.shape[0] / 10) == 1:
                    losses.append(sum(losses_buf) / len(losses_buf))
                    losses_buf = []

        return losses

    def predict(self, x: np.ndarray):
        assert self._input_size is not None, "Neural network must be trained first!"
        if x.ndim == 1:
            assert (
                self._input_size == 1
            ), f"""Input size differs from training! 
                    current = 1, during training = {self._input_size}"""
        else:
            assert (
                self._input_size == x.shape[-1]
            ), f"""Input size differs from training! 
                    current = {x.shape[-1]}, during training = {self._input_size}"""

        x = x / self._x_norm_coef

        result = []
        if x.ndim == 2 or (x.ndim == 1 and self._input_size == 1):
            for inp in x:
                result.append(self._forward(inp))
        else:
            result = self._forward(x)

        return np.array(result) * self._y_norm_coef

    def _forward(self, inp: np.ndarray, train: bool = False) -> np.ndarray:
        assert inp.ndim == 1 or inp.ndim == 0

        if train:
            self._neurons_outputs = [inp]

        last_layer_out = inp.reshape(-1, 1)

        for layer in self._layers:
            last_layer_out = layer.forward(last_layer_out)
            if train:
                self._neurons_outputs.append(last_layer_out)

        return last_layer_out.reshape(-1)

    def _backward(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        dl_dout = self._loss_derivative(actual, predicted)

        for i in range(len(self._layers) - 1, -1, -1):
            dl_dout = self._layers[i].backward(dl_dout, self._neurons_outputs[i])

    def _update_weights(self) -> None:
        for i in range(len(self._layers) - 1, -1, -1):
            self._layers[i].update_weights()

    def _init_knowing_sizes(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim == 1:
            self._input_size = 1
        else:
            self._input_size = x.shape[1]
        first_layer_size = self._layers[0].biases.size
        self._layers[0].weights = (
            np.random.rand(first_layer_size, self._input_size) / 10
        )
        self._layers[0].dl_dw = np.zeros_like(self._layers[0].weights)

        if y.ndim == 1:
            self._output_size = 1
        else:
            self._output_size = y.shape[1]
        last_layer_size = self._layers[-1].biases.size
        self._layers.append(
            Layer(last_layer_size, self._output_size, self._learning_rate, Linear())
        )


# dataset
diamonds_df = pd.read_csv("diamonds.csv")
features = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
target = ["price"]
cut_transform = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
clarity_transform = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}
color_transorm = {"D": 0, "E": 1, "F": 2, "G": 3, "H": 4, "I": 5, "J": 6}
diamonds_df["cut"] = diamonds_df["cut"].apply(lambda x: cut_transform.get(x))
diamonds_df["color"] = diamonds_df["color"].apply(lambda x: color_transorm.get(x))
diamonds_df["clarity"] = diamonds_df["clarity"].apply(
    lambda x: clarity_transform.get(x)
)
X = diamonds_df[features].copy().values
y = diamonds_df[target].values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=47, test_size=0.3
)


# Generate a dataset
# x = np.linspace((0, 0), (1000, 1000), 1000)
# y = 2 * x * x + 1

# Create an MLPRegressor object
regressor = MLPRegressor(
    hidden_layer_sizes=(1,),
    learning_rate=1e-4,
    max_iter=20,
    activation=Linear(),  # LeakedReLu(0.1)
    batch=16,
)

# Train the regressor
losses = regressor.train(X_train, y_train)
plt.plot(np.array(losses))

# losses = regressor.train(x, y)
# plt.plot(np.array(losses)[:, 0])
# plt.plot(np.array(losses)[:, 1])
plt.show()

# Predict the values of y for the given values of x
predicted_y = regressor.predict(X_test)

# Plot the results
plt.figure()
# plt.plot(x, y, label="Actual")
# plt.plot(x, predicted_y.reshape(-1), label="Predicted")
plt.legend()
plt.show()
input()
