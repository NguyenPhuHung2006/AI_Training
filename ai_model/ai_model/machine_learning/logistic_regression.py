import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, n_features):
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def get_parameters(self):
        return self.w, self.b
    
    def clear_parameters(self):
        self.w = np.zeros_like(self.w)
        self.b = 0
    
    def check_shape(self, X, y):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"the input expected {self.w.shape[0]} features, got {X.shape[1]}" 
            )
        
        if y.shape[1] != 1:
            raise ValueError(
                f"the output expected 1 feature, got {y.shape[1]}" 
            )
            
    def compute_accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def compute_cost(self, X, y, lambda_):
        y_pred = self.forward(X)
        eps = 1e-8
        m = X.shape[0]
        cost = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        reg = (lambda_ * np.sum(self.w**2)) / (2 * m)
        return cost + reg
            
    def predict(self, X, threshold=0.5):
        y_pred = self.sigmoid(X @ self.w + self.b)
        if threshold is not None:
            return (y_pred >= threshold).astype(int)
        return y_pred
    
    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)
            
    def gradient_descent(self, X, y, lr, lambda_):
        y_pred = self.forward(X)

        m = X.shape[0]   
        dw = (X.T @ (y_pred - y) + lambda_ * self.w) / m
        db = (np.sum(y_pred - y)) / m
        
        self.w -= lr * dw
        self.b -= lr * db

    def fit(self, X, y, n_iteration=10000, lr=0.01, lambda_=0, print_accuracy=True, check_every=1000):
        self.check_shape(X, y)
        
        for i in range(n_iteration):
            self.gradient_descent(X, y, lr, lambda_)
            
            if print_accuracy and i > 0 and i % max(1, check_every) == 0:
                accuracy = self.compute_accuracy(X, y)
                print(f"{i} | acc:{accuracy * 100:.2f}")
                
    def fit_with_validation(self, X, y, test_size=0.2, random_state=42, 
                            n_iteration=10000, lr=0.01, lambda_=0, 
                            print_accuracy=True, check_every=1000):
        self.check_shape(X, y)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
        for i in range(n_iteration):
            self.gradient_descent(X_train, y_train, lr, lambda_)
            
            if print_accuracy and i > 0 and i % max(1, check_every) == 0:
                train_accuracy = self.compute_accuracy(X_train, y_train)
                validation_accuracy = self.compute_accuracy(X_val, y_val)
                print(f"{i} | train:{train_accuracy * 100:.2f} | validation:{validation_accuracy * 100:.2f}")
        