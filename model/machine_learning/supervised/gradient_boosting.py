import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class GB_Base(ABC):
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def __init__(self, loss, learning_rate: float, max_depth, min_child_weight, lambda_=0, gamma=0, n_trees=100):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.gamma = gamma
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
    def leaf_value(self, gradients, hessians, lambda_):
        pass
    
    @abstractmethod
    def evaluate(self, y, y_pred):
        pass
    
    def gain(self, G_left, H_left, G_right, H_right, G_total, H_total, lambda_, eps=1e-15):

        left_score = (G_left ** 2) / (H_left + lambda_ + eps)
        right_score = (G_right ** 2) / (H_right + lambda_ + eps)
        parent_score = (G_total ** 2) / (H_total + lambda_ + eps)

        return 0.5 * (left_score + right_score - parent_score) - self.gamma
    
    def compute_score(self, x, gradients, hessians, lambda_):
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        g_sorted = gradients[sorted_idx]
        h_sorted = hessians[sorted_idx]

        G_total = np.sum(g_sorted)
        H_total = np.sum(h_sorted)

        G_left = 0.0
        H_left = 0.0

        best_gain = -float("inf")
        best_threshold = None
        
        for i in range(len(x_sorted) - 1):
            G_left += g_sorted[i]
            H_left += h_sorted[i]
            
            if x_sorted[i] == x_sorted[i + 1]:
                continue
            
            G_right = G_total - G_left
            H_right = H_total - H_left
            
            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue
            
            gain = self.gain(
                G_left, H_left, 
                G_right, H_right, 
                G_total, H_total, 
                lambda_
            )
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                
        return best_gain, best_threshold

    
    def best_split(self, X, gradients, hessians, lambda_):
        best_feature = None
        best_threshold = None
        best_gain = -float("inf")

        n_features = X.shape[1]

        for feature in range(n_features):
            gain, threshold = self.compute_score(
                X[:, feature], gradients, hessians, lambda_
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold, best_gain
    
    def build_decision_tree(self, X, gradients, hessians, lambda_, depth=0):
        H = np.sum(hessians)
        if depth >= self.max_depth or H < self.min_child_weight:
            return self.Node(value=self.leaf_value(gradients, hessians, lambda_))
        
        feature, threshold, gain = self.best_split(X, gradients, hessians, lambda_)
        
        # in case there's a bug in the code
        if feature is None or threshold is None or gain < 0:
            return self.Node(value=self.leaf_value(gradients, hessians, lambda_))
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self.Node(value=self.leaf_value(gradients, hessians, lambda_))
        
        left = self.build_decision_tree(
            X[left_mask], 
            gradients[left_mask], 
            hessians[left_mask], 
            lambda_, 
            depth + 1
        )
        
        right = self.build_decision_tree(
            X[right_mask], 
            gradients[right_mask], 
            hessians[right_mask], 
            lambda_, 
            depth + 1
        )
        
        return self.Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def _predict_tree(self, x, node: Node):
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def predict_tree(self, X, node: Node):
        return np.array([self._predict_tree(x, node) for x in X])
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_pred, dtype=float)
        
        for tree in self.trees:
            y_pred += self.learning_rate * self.predict_tree(X, tree)
            
        return y_pred
    
    def fit(self, X, y, print_every=None):
        self.trees = []
        self.init_pred = self.init_prediction(y)
        y_pred = np.full_like(y, self.init_pred, dtype=float)
        
        for i in range(self.n_trees):
            gradients = self.gradient(y, y_pred)
            hessians = self.hessian(y, y_pred)

            tree = self.build_decision_tree(X, gradients, hessians, self.lambda_)
            update = self.predict_tree(X, tree)
            y_pred += self.learning_rate * update
            self.trees.append(tree)
            
            if i > 0 and print_every is not None and i % max(1, print_every) == 0:
                print(f"{i:03d} / {self.n_trees:04d}")
    
    def fit_with_validation(
        self, X, y,
        test_size=0.2,
        random_state=42,
        early_stopping_rounds=10,
        check_every=1
    ): 
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        self.trees = []

        self.init_pred = self.init_prediction(y_train)

        y_pred_train = np.full_like(y_train, self.init_pred, dtype=float)
        y_pred_val   = np.full_like(y_val, self.init_pred, dtype=float)

        best_val_loss = float("inf")
        best_iter = 0
        rounds_no_improve = 0
        
        check_every = max(1, check_every)

        for i in range(self.n_trees):

            gradients = self.gradient(y_train, y_pred_train)
            hessians  = self.hessian(y_train, y_pred_train)

            tree = self.build_decision_tree(X_train, gradients, hessians, self.lambda_)

            train_update = self.predict_tree(X_train, tree)
            val_update   = self.predict_tree(X_val, tree)

            y_pred_train += self.learning_rate * train_update
            y_pred_val   += self.learning_rate * val_update

            self.trees.append(tree)

            if i % check_every == 0:

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
        return np.mean(y)
    
    def gradient(self, y, y_pred):
        return y_pred - y   

    def hessian(self, y, y_pred):
        return np.ones_like(y)

    def leaf_value(self, gradients, hessians, lambda_):
        H = np.sum(hessians)
        if H < 1e-6:
            H = 1e-6
        return - np.sum(gradients) / (H + lambda_)
    
    def evaluate(self, y, y_pred):
        return 0.5 * np.mean((y - y_pred)**2)
    
class GB_Classification(GB_Base):
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def init_prediction(self, y):
        mean_y = np.mean(y)
        eps = 1e-15
        mean_y = np.clip(mean_y, eps, 1 - eps)
        return np.log(mean_y / (1 - mean_y))

    def gradient(self, y, y_pred):
        p = self.sigmoid(y_pred)
        return p - y

    def hessian(self, y, y_pred):
        p = self.sigmoid(y_pred)
        return p * (1 - p)

    def leaf_value(self, gradients, hessians, lambda_):
        H = np.sum(hessians)
        if H < 1e-6:
            H = 1e-6
        return - np.sum(gradients) / (H + lambda_)
    
    def predict_label(self, X):
        y_pred = super().predict(X)
        proba = self.sigmoid(y_pred)
        return (proba >= 0.5).astype(int)
    
    def evaluate(self, y, y_pred):
        p = self.sigmoid(y_pred)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))