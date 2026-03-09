import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd

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
        exp_z = np.exp(z)
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
                
                msg.append(f"cost -> train:{train_loss:.4f}, validation:{val_loss:.4f}")
                if train_acc is not None and val_acc is not None:
                    msg.append(f"acc -> train:{train_acc * 100:.2f}%, validation:{val_acc * 100:.2f}%")

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
            
class GB_Base(ABC):
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, default_left=True):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.default_left = default_left
    
    def __init__(self, 
                 learning_rate: float, 
                 max_depth, 
                 min_child_weight, 
                 min_samples_leaf, 
                 n_classes=1,
                 sub_sample_size=0.8,
                 sub_feature_size=0.8,
                 lambda_=0, 
                 gamma=0, 
                 alpha=0, 
                 n_trees=100
                ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.n_classes = n_classes
        self.sub_sample_size = sub_sample_size
        self.sub_feature_size = sub_feature_size
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.n_trees = n_trees
        self.trees = []
        self.init_pred = None
    
    @abstractmethod
    def init_prediction(self, y):
        pass
    
    @abstractmethod
    def gradient(self, y, y_pred):
        pass
    
    @abstractmethod
    def hessian(self, y, y_pred):
        pass
    
    @abstractmethod
    def leaf_value(self, gradients, hessians):
        pass
    
    @abstractmethod
    def evaluate(self, y, y_pred):
        pass
    
    def soft_threshold(self, G):
        if G > self.alpha:
            return G - self.alpha
        elif G < -self.alpha:
            return G + self.alpha
        return 0
    
    def gain(self, G_left, H_left, G_right, H_right, G_total, H_total, eps=1e-15):

        GL = self.soft_threshold(G_left)
        GR = self.soft_threshold(G_right)
        GP = self.soft_threshold(G_total)

        left_score = (GL ** 2) / (H_left + self.lambda_ + eps)
        right_score = (GR ** 2) / (H_right + self.lambda_ + eps)
        parent_score = (GP ** 2) / (H_total + self.lambda_ + eps)

        return 0.5 * (left_score + right_score - parent_score) - self.gamma
    
    def compute_score(self, x, gradients, hessians):
        
        nan_mask = np.isnan(x)
        not_nan_mask = ~nan_mask
        
        x_valid = x[not_nan_mask]
        g_valid = gradients[not_nan_mask]
        h_valid = hessians[not_nan_mask]

        g_nan = gradients[nan_mask]
        h_nan = hessians[nan_mask]

        G_nan = np.sum(g_nan)
        H_nan = np.sum(h_nan)

        if len(x_valid) == 0:
            return -float("inf"), None, True
        
        sorted_idx = np.argsort(x_valid)
        x_sorted = x_valid[sorted_idx]
        g_sorted = g_valid[sorted_idx]
        h_sorted = h_valid[sorted_idx]

        G_total = np.sum(g_sorted) + G_nan
        H_total = np.sum(h_sorted) + H_nan
        
        G_valid_total = np.sum(g_sorted)
        H_valid_total = np.sum(h_sorted)

        G_left = 0.0
        H_left = 0.0

        best_gain = -float("inf")
        best_threshold = None
        best_default_left = True
        
        for i in range(len(x_sorted) - 1):

            G_left += g_sorted[i]
            H_left += h_sorted[i]
            
            if x_sorted[i] == x_sorted[i + 1]:
                continue

            G_right = G_valid_total - G_left
            H_right = H_valid_total - H_left
            
            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue                    
            
            gain_left = self.gain(
                G_left + G_nan,
                H_left + H_nan,
                G_right,
                H_right,
                G_total,
                H_total
            )
            
            gain_right = self.gain(
                G_left,
                H_left,
                G_right + G_nan,
                H_right + H_nan,
                G_total,
                H_total
            )

            if gain_left > best_gain:
                best_gain = gain_left
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                best_default_left = True

            if gain_right > best_gain:
                best_gain = gain_right
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                best_default_left = False
                
        return best_gain, best_threshold, best_default_left
    
    def get_random_indices(self, ratio, total_size, replace=False):
        size = max(1, int(ratio * total_size))
        return np.random.choice(total_size, size, replace=replace)
    
    def best_split(self, X, gradients, hessians):
        best_feature = None
        best_threshold = None
        best_gain = -float("inf")
        best_default_left = True

        n_features = X.shape[1]
        
        features = self.get_random_indices(self.sub_feature_size, n_features, replace=False)

        for feature in features:
            gain, threshold, default_left = self.compute_score(
                X[:, feature], gradients, hessians
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_default_left = default_left

        return best_feature, best_threshold, best_gain, best_default_left
    
    def build_decision_tree(self, X, gradients, hessians, depth=0):
        H = np.sum(hessians)
        if depth >= self.max_depth or H < self.min_child_weight:
            return self.Node(value=self.leaf_value(gradients, hessians))
        
        feature, threshold, gain, default_left = self.best_split(X, gradients, hessians)
        
        # in case there's a bug in the code
        if feature is None or threshold is None or gain <= 0:
            return self.Node(value=self.leaf_value(gradients, hessians))
        
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        nan_mask = np.isnan(X[:, feature])

        if default_left:
            left_mask = left_mask | nan_mask
        else:
            right_mask = right_mask | nan_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self.Node(value=self.leaf_value(gradients, hessians))
        
        left = self.build_decision_tree(
            X[left_mask], 
            gradients[left_mask], 
            hessians[left_mask], 
            depth + 1
        )
        
        right = self.build_decision_tree(
            X[right_mask], 
            gradients[right_mask], 
            hessians[right_mask], 
            depth + 1
        )
        
        return self.Node(
            feature=feature, 
            threshold=threshold, 
            left=left, 
            right=right, 
            default_left=default_left
        )
    
    def _predict_tree(self, x, node: Node):
        while node.value is None:
            value = x[node.feature]

            if np.isnan(value):
                if node.default_left:
                    node = node.left
                else:
                    node = node.right

            elif value <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value
    
    def predict_tree(self, X, node: Node):
        return np.array([self._predict_tree(x, node) for x in X])
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.tile(self.init_pred, (n_samples, 1))
        
        for tree_round in self.trees:
            for k, tree in enumerate(tree_round):
                update = self.predict_tree(X, tree)
                y_pred[:, k] += self.learning_rate * update
        
        return y_pred
    
    def fit(self, X, y, check_every=None):
        if y.ndim == 1:
            y = y[:, None]
        self.trees = []
        self.init_pred = self.init_prediction(y)
        n_samples = X.shape[0]
        y_pred = np.tile(self.init_pred, (n_samples, 1))
        
        for i in range(self.n_trees):
            gradients = self.gradient(y, y_pred)
            hessians = self.hessian(y, y_pred)
            
            trees_round = []
            
            for k in range(self.n_classes):
                
                idx = self.get_random_indices(self.sub_sample_size, len(y), replace=False)
                X_sub = X[idx]
                g_sub = gradients[idx, k]
                h_sub = hessians[idx, k]

                tree = self.build_decision_tree(X_sub, g_sub, h_sub)
                update = self.predict_tree(X, tree)
                y_pred[:, k] += self.learning_rate * update

                trees_round.append(tree)

            self.trees.append(trees_round)
            
            if i > 0 and check_every is not None and i % max(1, check_every) == 0:
                print(f"{i:03d} / {self.n_trees:04d}")
    
    def fit_with_validation(
        self, X, y,
        test_size=0.2,
        random_state=42,
        early_stopping_rounds=10,
        check_every=1
    ): 
        if y.ndim == 1:
            y = y[:, None]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        self.trees = []
        self.init_pred = self.init_prediction(y_train)
        
        y_pred_train = np.tile(self.init_pred, (X_train.shape[0], 1))
        y_pred_val   = np.tile(self.init_pred, (X_val.shape[0], 1))

        best_val_loss = float("inf")
        best_iter = 0
        rounds_no_improve = 0
        
        for i in range(self.n_trees):

            gradients = self.gradient(y_train, y_pred_train)
            hessians  = self.hessian(y_train, y_pred_train)
            
            trees_round = []
            for k in range(self.n_classes):
                idx = self.get_random_indices(self.sub_sample_size, len(y_train), replace=False)
                X_sub = X_train[idx]
                g_sub = gradients[idx, k]
                h_sub = hessians[idx, k]

                tree = self.build_decision_tree(X_sub, g_sub, h_sub)
                train_update = self.predict_tree(X_train, tree)
                val_update = self.predict_tree(X_val, tree)
                y_pred_train[:, k] += self.learning_rate * train_update
                y_pred_val[:, k] += self.learning_rate * val_update

                trees_round.append(tree)

            self.trees.append(trees_round)
            
            if check_every is not None and i % max(1, check_every) == 0:

                val_loss = self.evaluate(y_val, y_pred_val)

                print(
                    f"{i:03d} | "
                    f"train:{self.evaluate(y_train, y_pred_train):.4f} | "
                    f"val:{val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iter = i
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if rounds_no_improve >= early_stopping_rounds:
                    print(f"Early stopping at iter {i}")
                    break

        self.trees = self.trees[:best_iter + 1]

        print(f"Best iteration: {best_iter}")
        print(f"Best val loss : {best_val_loss:.4f}")
    
class GB_Regression(GB_Base):
    
    def init_prediction(self, y):
        return np.mean(y, axis=0)
    
    def gradient(self, y, y_pred):
        return y_pred - y   

    def hessian(self, y, y_pred):
        return np.ones_like(y)

    def leaf_value(self, gradients, hessians):
        H = np.sum(hessians)
        G = np.sum(gradients)
        return - super().soft_threshold(G) / (H + self.lambda_)
    
    def evaluate(self, y, y_pred):
        return 0.5 * np.mean(np.sum((y - y_pred)**2, axis=1))
    
class GB_Classification(GB_Base):
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def init_prediction(self, y):
        mean_y = np.mean(y, axis=0)
        eps = 1e-15
        mean_y = np.clip(mean_y, eps, 1 - eps)
        return np.log(mean_y / (1 - mean_y))

    def gradient(self, y, y_pred):
        p = self.sigmoid(y_pred)
        return p - y

    def hessian(self, y, y_pred):
        p = self.sigmoid(y_pred)
        return p * (1 - p)

    def leaf_value(self, gradients, hessians):
        H = np.sum(hessians)
        G = np.sum(gradients)
        return - super().soft_threshold(G) / (H + self.lambda_)
    
    def predict_label(self, X):
        probs = self.predict_prob(X)
        return (probs >= 0.5).astype(int)
    
    def predict_prob(self, X):
        y_pred = super().predict(X)
        return self.sigmoid(y_pred)
    
    def evaluate(self, y, y_pred):
        p = self.sigmoid(y_pred)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.mean(np.sum(loss, axis=1))
    
class GB_MultiClassification(GB_Base):
    
    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def init_prediction(self, y):
        y = y.ravel()
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / np.sum(counts)
        probs = np.clip(probs, 1e-15, 1)
        return np.log(probs)
    
    def gradient(self, y, y_pred):
        y = y.ravel()
        p = self.softmax(y_pred)
        y_onehot = np.zeros_like(p)
        y_onehot[np.arange(len(y)), y] = 1
        return p - y_onehot
    
    def hessian(self, y, y_pred):
        p = self.softmax(y_pred)
        h = p * (1 - p)
        return np.maximum(h, 1e-6)
    
    def leaf_value(self, gradients, hessians):
        H = np.sum(hessians)
        G = np.sum(gradients)
        return - super().soft_threshold(G) / (H + self.lambda_)
    
    def predict_prob(self, X):
        y_pred = super().predict(X)
        return self.softmax(y_pred)

    def predict_label(self, X):
        return np.argmax(self.predict_prob(X), axis=1, keepdims=True)
    
    def evaluate(self, y, y_pred):
        y = y.ravel()
        p = self.softmax(y_pred)

        eps = 1e-15
        p = np.clip(p, eps, 1)

        log_probs = -np.log(p[np.arange(len(y)), y])
        return np.mean(log_probs)
    
def fill_na_median(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())
            
    return df

def process_dates(df):
    df = df.copy()
    
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], errors="coerce")

    df = df.assign(
        search_month = df["date_time"].dt.month,
        stay_days = (df["srch_co"] - df["srch_ci"]).dt.days,
        booking_lead = (df["srch_ci"] - df["date_time"]).dt.days
    )

    df = df.drop(["date_time", "srch_ci", "srch_co"], axis=1)

    return df
    
def read_data(train_path, test_path, dest_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_dest = pd.read_csv(dest_path)
    
    df_train = df_train.sample(n=100000, random_state=42)
    df_test = df_test.sample(n=10000, random_state=42)
    
    df_train = df_train.merge(df_dest, on="srch_destination_id", how="left")
    df_test = df_test.merge(df_dest, on="srch_destination_id", how="left")
    
    df_train = process_dates(df_train)
    df_test = process_dates(df_test)

    y_train = df_train["hotel_cluster"]

    drop_cols = ["hotel_cluster", "is_booking", "cnt"]

    df_train = df_train.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols, errors="ignore")

    df_train = fill_na_median(df_train)
    df_test = fill_na_median(df_test)
    
    X_train = df_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = df_test.to_numpy()
        
    return X_train, y_train, X_test
    
            
def main():
    X_train, y_train, X_test = read_data("new_dataset/train.csv", "new_dataset/test.csv", "new_dataset/destinations.csv")
        
if __name__ == '__main__':
    main()
    print("completed")
