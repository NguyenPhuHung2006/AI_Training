import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomDecisionTree:
    def __init__(self, max_depth, min_samples):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        
    def binary_entropy(self, p):
        if p == 0 or p == 1:
            return 0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    
    def most_frequent_value(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)].astype(bool)
    
    def compute_score(self, x, y):
        thresholds = np.unique(x)
        n_thresholds = len(thresholds)
        n_samples = len(y)
        best_score, best_threshold = float("inf"), None
            
        for i in range(n_thresholds - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            left = y[x <= threshold]
            right = y[x > threshold]
            n_left = len(left)
            n_right = len(right)
            
            if n_left == 0 or n_right == 0:
                continue
            
            w_left = n_left / n_samples
            w_right = n_right / n_samples
            
            # y values are 0 and 1
            p_left = left.mean()
            p_right = right.mean()
            
            score = w_left * self.binary_entropy(p_left) + w_right * self.binary_entropy(p_right)
            if score < best_score:
                best_score = score
                best_threshold = threshold
                
        return best_score, best_threshold
    
    def best_split(self, X, y, feature_indices):
        best_feature = None
        best_threshold = None
        best_score = float("inf")

        for feature in feature_indices:
            score, threshold = self.compute_score(X[:, feature], y)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold
    
    def random_features_indices(self, n_features):
        m = int(np.sqrt(n_features))
        return np.random.choice(n_features, m, replace=False)
    
    def build_decision_tree(self, X, y, depth=0):
        if (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples or len(np.unique(y)) == 1:
            return Node(value=self.most_frequent_value(y))
        
        n_features = X.shape[1]
        feature_indices = self.random_features_indices(n_features)
        
        feature, threshold = self.best_split(X, y, feature_indices)
        
        # in case there's a bug in the code
        if feature is None or threshold is None:
            return Node(value=self.most_frequent_value(y))
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left = self.build_decision_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_decision_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y):
        unique = np.unique(y)
        if not set(unique).issubset({0,1}):
            raise ValueError("Only binary labels {0,1} are supported.")

        self.root = self.build_decision_tree(X, y)
        
    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)
    
    def predict(self, x):
        return self._predict(x, self.root)
    
    def predict_batch(self, X):
        if self.root is None:
            raise ValueError("The tree has not been trained yet.") 
        return np.array([self.predict(x) for x in X])
    
class RandomForest:
    def __init__(self, max_depth, min_samples, n_trees):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_trees = n_trees
        self.forest = []
        
    def random_sample(self, X, y):
        n = len(X)
        idx = np.random.choice(n, size=n, replace=True)
        return X[idx], y[idx]
    
    def fit(self, X, y, check_every=None):
        self.forest = []
        for i in range(self.n_trees):
            if i > 0 and check_every is not None and i % max(1, check_every) == 0:
                print(f"{i:.3d} / {self.n_trees}")
            
            Xb, yb = self.random_sample(X, y)
            tree = RandomDecisionTree(self.max_depth, self.min_samples)
            tree.fit(Xb, yb)
            
            self.forest.append(tree)
            
    def predict(self, x):
        votes = [tree.predict(x) for tree in self.forest]
        return Counter(votes).most_common(1)[0][0].astype(bool)
    
    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])
    
    def fit_with_validation(self, X, y, test_size=0.2, random_state=42, check_every=None):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        self.fit(X_train, y_train, check_every=check_every)
        
        y_train_predict = self.predict_batch(X_train)
        y_val_predict = self.predict_batch(X_val)
        
        print(f"accuracy | train:{np.mean(y_train_predict == y_train) * 100:.2f} | validation: {np.mean(y_val_predict == y_val) * 100:.2f}")
    
    