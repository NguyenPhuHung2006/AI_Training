import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


# =========================
# Cost functions
# =========================
class Cost(ABC):

    @abstractmethod
    def compute(self, y_pred, y):
        pass

    @abstractmethod
    def derivative(self, y_pred, y):
        pass


class MSE(Cost):
    def compute(self, y_pred, y):
        return np.mean((y_pred - y) ** 2) / 2

    def derivative(self, y_pred, y):
        return (y_pred - y)


class BinaryCrossEntropy(Cost):
    def compute(self, y_pred, y):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred)
        )

    def derivative(self, y_pred, y):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y / y_pred) + ((1 - y) / (1 - y_pred))


# =========================
# Activations
# =========================
class Activation(ABC):

    @abstractmethod
    def compute(self, z):
        pass

    @abstractmethod
    def compute_derivative(self, z):
        pass


class Sigmoid(Activation):
    def compute(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_derivative(self, z):
        s = self.compute(z)
        return s * (1 - s)


class Linear(Activation):
    def compute(self, z):
        return z

    def compute_derivative(self, z):
        return np.ones_like(z)


class Tanh(Activation):
    def compute(self, z):
        z = np.clip(z, -500, 500)
        return np.tanh(z)

    def compute_derivative(self, z):
        t = self.compute(z)
        return 1 - t**2


class Relu(Activation):
    def compute(self, z):
        return np.maximum(z, 0)

    def compute_derivative(self, z):
        return (z > 0).astype(float)


# =========================
# Layer
# =========================
class Layer:
    def __init__(self, n_units, activation, n_inputs):
        self.n_units = n_units
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_outputs = n_units

        # He vs Xavier init
        if isinstance(activation, Relu):
            scale = np.sqrt(2 / n_inputs)
        else:
            scale = np.sqrt(1 / n_inputs)

        self.W = np.random.randn(n_units, n_inputs) * scale
        self.b = np.zeros((n_units, 1))

        # cache for backprop
        self.Z = None
        self.A = None
        self.input = None

    def forward(self, X, store=True):
        Z = self.W @ X + self.b
        A = self.activation.compute(Z)

        if store:
            self.input = X
            self.Z = Z
            self.A = A

        return A


# =========================
# Neural Network
# =========================
class NN:
    def __init__(self, n_inputs, cost_function=None):
        self.n_inputs = n_inputs
        self.cost_function = cost_function
        self.layers = []

    def add_layer(self, n_units, activation):
        n_inputs = self.layers[-1].n_outputs if self.layers else self.n_inputs
        self.layers.append(Layer(n_units, activation, n_inputs))

    # forward pass
    def predict(self, X, store=True):
        if not self.layers:
            raise ValueError("NN doesn't have any layers")

        if X.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected input with {self.n_inputs} features, got {X.shape[0]}"
            )

        A = X
        for layer in self.layers:
            A = layer.forward(A, store)

        return A

    # accuracy metric
    def accuracy(self, y_pred, y, threshold=0.5):
        preds = (y_pred > threshold).astype(int)
        return np.mean(preds == y)

    # evaluation helper
    def evaluate(self, X, y):
        y_pred = self.predict(X, store=False)
        loss = self.cost_function.compute(y_pred, y)

        acc = None
        if isinstance(self.cost_function, BinaryCrossEntropy):
            acc = self.accuracy(y_pred, y)

        return loss, acc


    # backward pass
    def _backward(self, y_pred, y, lr):
        m = y.shape[1]
        dA = self.cost_function.derivative(y_pred, y)

        for layer in reversed(self.layers):
            dZ = dA * layer.activation.compute_derivative(layer.Z)
            dW = (dZ @ layer.input.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = layer.W.T @ dZ

            layer.W -= lr * dW
            layer.b -= lr * db

    # training
    def fit(self, X, y, n_iterations=10000, lr=0.01,
            print_cost=False, print_accuracy=False):

        if not self.cost_function:
            raise ValueError("Cost function must be defined")

        print_every = max(1, n_iterations // 10)

        for i in range(n_iterations):
            y_pred = self.predict(X, store=True)
            self._backward(y_pred, y, lr)

            if i % print_every == 0 and (print_cost or print_accuracy):
                
                msg = [f"{i} |"]
                
                loss = self.cost_function.compute(y_pred, y)
                
                if print_cost:
                    msg.append(f" loss:{loss:.4f}")

                if print_accuracy and isinstance(self.cost_function, BinaryCrossEntropy):
                    acc = self.accuracy(y_pred, y)
                    msg.append(f" acc:{acc * 100:.2f}%")

                print(''.join(msg))

    # train with validation split
    def fit_with_validation(self, X, y,
                            n_iterations=10000,
                            lr=0.01,
                            print_cost=True,
                            print_accuracy=False,
                            test_size=0.2,
                            random_state=42):

        X_train, X_val, y_train, y_val = train_test_split(
            X.T, y.T, test_size=test_size, random_state=random_state
        )

        X_train, X_val = X_train.T, X_val.T
        y_train, y_val = y_train.T, y_val.T

        print_every = max(1, n_iterations // 10)

        for i in range(n_iterations):
            y_pred = self.predict(X_train, store=True)
            self._backward(y_pred, y_train, lr)

            if i % print_every == 0 and (print_cost or print_accuracy):
                train_loss, train_acc = self.evaluate(X_train, y_train)
                val_loss, val_acc = self.evaluate(X_val, y_val)
                
                msg = [f"{i}\n"]
                
                if print_cost:
                    msg.append(f"cost -> train:{train_loss:.4f}, validation:{val_loss:.4f}")
                    
                if print_accuracy and isinstance(self.cost_function, BinaryCrossEntropy):
                    msg.append(f"acc -> train:{train_acc * 100:.2f}, validation:{val_acc * 100:.2f}")

                print(''.join(msg))
