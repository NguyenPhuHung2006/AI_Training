import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
                
def read_data(path: str):
    df = pd.read_csv(path)

    passenger_id = None
    if "PassengerId" in df.columns:
        passenger_id = df["PassengerId"]
        df = df.drop(columns=["PassengerId"])

    if "Name" in df.columns:
        df = df.drop(columns=["Name"])

    return df, passenger_id

def preprocess_common(df):
    df = df.copy()

    # Boolean 
    binary_cols = ["CryoSleep", "VIP"]
    df[binary_cols] = df[binary_cols].astype("object").fillna("Unknown")
    df = pd.get_dummies(df, columns=binary_cols, drop_first=True)

    # Cabin
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0].fillna("Unknown")
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2].fillna("Unknown")
    df = df.drop(columns=["Cabin"])
    
    # Categorical
    cat_cols = ["HomePlanet", "Destination", "Deck", "Side"]
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    
    df[spend_cols] = df[spend_cols].fillna(0)
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    
    return df

def fit_preprocess(df, normalized):
    
    df = preprocess_common(df)
    
    stats = {}

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpend"]
    num_cols = ["Age", "CabinNum"]
    
    num_medians = df[num_cols].median()
    
    df[num_cols] = df[num_cols].fillna(num_medians)
    
    stats["columns"] = df.columns
    stats["num_cols"] = num_cols
    stats["num_medians"] = num_medians
    
    if normalized:
        num_means = df[num_cols].mean()
        num_stds = df[num_cols].std() + 1e-8
        df[num_cols] = (df[num_cols] - num_means) / num_stds
        df[spend_cols] = np.log1p(df[spend_cols])
        stats["spend_cols"] = spend_cols
        stats["num_means"] = num_means
        stats["num_stds"] = num_stds

    return df, stats

def apply_preprocess(df, stats, normalized):
    
    df = preprocess_common(df)
    
    num_cols = stats["num_cols"]
    num_medians = stats["num_medians"]
    df[num_cols] = df[num_cols].fillna(num_medians)
    
    if normalized:
        spend_cols = stats["spend_cols"]
        num_means = stats["num_means"]
        num_stds = stats["num_stds"]
        df[spend_cols] = np.log1p(df[spend_cols])
        df[num_cols] = (df[num_cols] - num_means) / num_stds

    # Align columns
    df = df.reindex(columns=stats["columns"], fill_value=0)

    return df

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

def get_validation(X_train, y_train, model):
    model.fit_with_validation(X_train.T, y_train.T, print_cost=False, print_accuracy=True, n_iterations=20000)
    
def get_result(X_train, y_train, X_test, passenger_id, model):
    
    model.fit(X_train.T, y_train.T, print_accuracy=True)
    
    y_test = model.predict(X_test.T)
    y_test = y_test >= 0.5
    y_test = y_test.reshape(-1)
    
    df_out = pd.DataFrame({
        "PassengerId": passenger_id,
        "Transported": y_test
    })

    df_out.to_csv("outputs/neural_network.csv", index=False)
    

def main():
    X_train_raw, _ = read_data("data/train.csv")
    y_train = X_train_raw["Transported"].astype(float).to_numpy().reshape(-1, 1)
    X_train_raw = X_train_raw.drop(columns=["Transported"])
    
    X_test_raw, passenger_id = read_data("data/test.csv")
    
    normalized = True

    X_train_df, stats = fit_preprocess(X_train_raw, normalized)
    X_test_df = apply_preprocess(X_test_raw, stats, normalized)
    
    X_train = X_train_df.to_numpy(dtype=float)
    X_test  = X_test_df.to_numpy(dtype=float)
    
    model = NN(X_train.shape[1], BinaryCrossEntropy())
    model.add_layer(32, Relu())
    model.add_layer(16, Relu())
    model.add_layer(y_train.shape[1], Sigmoid())
    
    # get_validation(X_train, y_train, model)
    get_result(X_train, y_train, X_test, passenger_id, model)

if __name__ == "__main__":
    main()
    print("completed")

        
