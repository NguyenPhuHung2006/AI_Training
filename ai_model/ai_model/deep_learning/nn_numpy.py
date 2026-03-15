import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

# =========================
# Activations
# =========================
class Activation(ABC):

    @abstractmethod
    def compute(self, z):
        pass
    
    @abstractmethod
    def derivative_from_a(self, a):
        pass
    
    @abstractmethod
    def derivative_from_z(self, z):
        pass


class Sigmoid(Activation):
    def compute(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def derivative_from_a(self, a):
        return a * (1 - a)

    def derivative_from_z(self, z):
        s = self.compute(z)
        return self.derivative_from_a(s)


class Linear(Activation):
    def compute(self, z):
        return z
    
    def derivative_from_a(self, a):
        return np.ones_like(a)
    
    def derivative_from_z(self, z):
        return np.ones_like(z)


class Tanh(Activation):
    def compute(self, z):
        z = np.clip(z, -500, 500)
        return np.tanh(z)
    
    def derivative_from_a(self, a):
        return 1 - a**2
    
    def derivative_from_z(self, z):
        t = self.compute(z)
        return self.derivative_from_a(t)


class Relu(Activation):
    def compute(self, z):
        return np.maximum(z, 0)
    
    def derivative_from_a(self, a):
        return (a > 0).astype(float)
    
    def derivative_from_z(self, z):
        return (z > 0).astype(float)
    
class Softmax(Activation):
    def compute(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(np.clip(z, -500, 500))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def derivative_from_a(self, a):
        return a * (1 - a)

    def derivative_from_z(self, z):
        a = self.compute(z)
        return self.derivative_from_a(a)
    
# =========================
# Cost functions
# =========================
class Cost(ABC):

    @abstractmethod
    def compute_cost(self, y_pred, y):
        pass

    @abstractmethod
    def compute_dA(self, y_pred, y):
        pass
    
    def compute_dZ(self, dA, Y, layer):
        return dA * layer.activation.derivative_from_a(layer.A)


class MSE(Cost):
    def compute_cost(self, y_pred, y):
        return 0.5 * np.mean(np.sum((y - y_pred)**2, axis=1))

    def compute_dA(self, y_pred, y):
        return (y_pred - y)


class BCE(Cost):
    def compute_cost(self, y_pred, y):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return np.mean(np.sum(loss, axis=1))

    def compute_dA(self, y_pred, y):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y / y_pred) + ((1 - y) / (1 - y_pred))
    
    def compute_dZ(self, dA, Y, layer):
        if isinstance(layer.activation, Sigmoid):
            return layer.A - Y
        return super().compute_dZ(dA, Y, layer)
    
    def compute_accuracy(self, y_pred, y, threshold=0.5):
        preds = (y_pred > threshold).astype(int)
        return np.mean(preds == y)
    
class CCE(Cost):
    def compute_cost(self, y_pred, y):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))

    def compute_dA(self, y_pred, y):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1)
        return -(y / y_pred)

    def compute_dZ(self, dA, Y, layer):
        if isinstance(layer.activation, Softmax):
            return layer.A - Y
        return super().compute_dZ(dA, Y, layer)
    
    def compute_accuracy(self, y_pred, y):
        preds = np.argmax(y_pred, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)

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

        self.W = np.random.randn(n_inputs, n_units) * scale
        self.b = np.zeros((1, n_units))

        # cache for backprop
        self.Z = None
        self.A = None
        self.input = None

    def forward(self, X, store=True):
        Z = X @ self.W + self.b
        A = self.activation.compute(Z)

        if store:
            self.input = X
            self.Z = Z
            self.A = A

        return A
    
    def backward(self, dA, Y, cost_function, lr, m):     
        dZ = cost_function.compute_dZ(dA, Y, self)
        dW = (self.input.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA = dZ @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db
        
        return dA


# =========================
# Neural Network
# =========================
class NeuralNetwork:
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

        if X.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected input with {self.n_inputs} features, got {X.shape[1]}"
            )

        A = X
        for layer in self.layers:
            A = layer.forward(A, store)

        return A
    
    # backward pass
    def backward(self, y_pred, y, lr):
        m = y.shape[0]
        dA = self.cost_function.compute_dA(y_pred, y)

        for layer in reversed(self.layers):
            dA = layer.backward(dA, y, self.cost_function, lr, m)

    # evaluation helper
    def evaluate(self, y_pred, y):
        loss = self.cost_function.compute_cost(y_pred, y)
        acc = None
        if isinstance(self.cost_function, BCE) or isinstance(self.cost_function, CCE):
            acc = self.cost_function.compute_accuracy(y_pred, y)

        return loss, acc

    # training
    def fit(self, X, y, n_iterations=10000, lr=0.01, check_every=None):

        if not self.cost_function:
            raise ValueError("Cost function must be defined")

        for i in range(n_iterations):
            y_pred = self.predict(X, store=True)
            self.backward(y_pred, y, lr)

            if check_every is not None and i % max(1, check_every) == 0:
                
                msg = [f"{i} |"]
                    
                loss, acc = self.evaluate(y_pred, y)
                msg.append(f" loss:{loss:.4f}")
                if acc is not None:
                    msg.append(f" acc:{acc * 100:.2f}%")

                print(''.join(msg))

    # train with validation split
    def fit_with_validation(self, X, y,
                            n_iterations=10000,
                            lr=0.01,
                            check_every=None,
                            test_size=0.2,
                            random_state=42):

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        for i in range(n_iterations):
            y_pred_train = self.predict(X_train, store=True)
            self.backward(y_pred_train, y_train, lr)

            if check_every is not None and i % max(1, check_every) == 0:
                y_pred_val = self.predict(X_val, store=False)
                train_loss, train_acc = self.evaluate(y_pred_train, y_train)
                val_loss, val_acc = self.evaluate(y_pred_val, y_val)
                
                msg = [f"{i}\n"]
                
                msg.append(f"cost -> train:{train_loss:.4f}, validation:{val_loss:.4f} ")
                if train_acc is not None and val_acc is not None:
                    msg.append(f"| acc -> train:{train_acc * 100:.2f}%, validation:{val_acc * 100:.2f}%")

                print(''.join(msg))
                
    def save(self, filename):
        params = {}

        for i, layer in enumerate(self.layers):
            params[f"W{i}"] = layer.W
            params[f"b{i}"] = layer.b

        np.savez(filename, **params)
        
    def load(self, filename):
        data = np.load(filename)

        for i, layer in enumerate(self.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]