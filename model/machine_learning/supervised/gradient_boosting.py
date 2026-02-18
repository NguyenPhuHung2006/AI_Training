import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class GB_BaseLoss(ABC):
    
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
    
class GB_MSE(GB_BaseLoss):
    
    def init_prediction(self, y):
        return np.mean(y)
    
    def gradient(self, y, y_pred):
        return y_pred - y   

    def hessian(self, y, y_pred):
        return np.ones_like(y)

    def leaf_value(self, gradients, hessians):
        return - np.sum(gradients) / (np.sum(hessians) + 1e-15)
    
class GB_BCE(GB_BaseLoss):
    
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

    def leaf_value(self, gradients, hessians):
        return - np.sum(gradients) / (np.sum(hessians) + 1e-15)
    
class GB_Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class GB_DecisionTree:
    def __init__(self, max_depth, min_samples):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        
    def gain(self, g_left, h_left, g_right, h_right, g_total, h_total, lambda_):
        eps = 1e-15

        left_score = (g_left ** 2) / (h_left + lambda_ + eps)
        right_score = (g_right ** 2) / (h_right + lambda_ + eps)
        parent_score = (g_total ** 2) / (h_total + lambda_ + eps)

        return left_score + right_score - parent_score
    
    def compute_score(self, x, gradients, hessians, lambda_):
        thresholds = np.unique(x)
        best_gain = -float("inf")
        best_threshold = None

        g_total = np.sum(gradients)
        h_total = np.sum(hessians)

        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2

            left_mask = x <= threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            g_left = np.sum(gradients[left_mask])
            h_left = np.sum(hessians[left_mask])

            g_right = np.sum(gradients[right_mask])
            h_right = np.sum(hessians[right_mask])

            gain = self.gain(
                g_left, h_left,
                g_right, h_right,
                g_total, h_total,
                lambda_
            )

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

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

    
    def build_decision_tree(self, X, gradients, hessians, loss, lambda_, depth=0):
        if (self.max_depth and depth >= self.max_depth) or X.shape[0] <= self.min_samples:
            return GB_Node(value=loss.leaf_value(gradients, hessians))
        
        feature, threshold, gain = self.best_split(X, gradients, hessians, lambda_)
        
        # in case there's a bug in the code
        if feature is None or threshold is None or gain <= 0:
            return GB_Node(value=loss.leaf_value(gradients, hessians))
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left = self.build_decision_tree(
            X[left_mask], 
            gradients[left_mask], 
            hessians[left_mask], 
            loss, lambda_, depth + 1)
        
        right = self.build_decision_tree(
            X[right_mask], 
            gradients[right_mask], 
            hessians[right_mask], 
            loss, lambda_, depth + 1)
        
        return GB_Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y, loss: GB_BaseLoss, lambda_=0, y_pred=None, get_pred=True):
        if y_pred is None:
            y_init = loss.init_prediction(y)
            y_pred = np.full_like(y, y_init, dtype=float)
        
        gradients = loss.gradient(y, y_pred)
        hessians = loss.hessian(y, y_pred)
        
        self.root = self.build_decision_tree(X, gradients, hessians, loss, lambda_)
        
        if get_pred: 
            return self.predict(X)
                
    def _predict(self, x, node):
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
    
    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

class GradientBoosting:
    def __init__(self, loss, learning_rate: float):
        self.loss = loss
        self.learning_rate = learning_rate
        self.trees = []
        self.init_pred = None
        
    def fit(self, X, y, n_trees, max_depth, min_samples, lambda_=0, check_every=None):
        self.trees = []
        self.init_pred = self.loss.init_prediction(y)
        y_pred = np.full_like(y, self.init_pred, dtype=float)
        
        for i in range(n_trees):
            tree = GB_DecisionTree(max_depth, min_samples)
            new_pred = tree.fit(X, y, self.loss, lambda_, y_pred)
            y_pred += self.learning_rate * new_pred
            self.trees.append(tree)
            
            if i > 0 and check_every is not None and i % max(1, check_every) == 0:
                print(f"{i:.3d} / {n_trees}")
            
    def predict(self, X):
        y_pred = self.init_pred
        y_pred = np.full(X.shape[0], self.init_pred)
        
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
    
    
        