import numpy as np

class LogisticRegression:
    def __init__(self, n_features, n_outputs=1):
        self.W = np.zeros((n_features, n_outputs))
        self.b = np.zeros((1, n_outputs))
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def get_parameters(self):
        return self.W, self.b
    
    def clear_parameters(self):
        self.W = np.zeros_like(self.W)
        self.b = np.zeros_like(self.b)
    
    def check_shape(self, X, Y):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"the input expected {self.w.shape[0]} features, got {X.shape[1]}" 
            )
        
        if Y.shape[1] != self.b.shape[1]:
            raise ValueError(
                f"the input has {self.b.shape[1]} features, got {Y.shape[0]}" 
            )
            
    def predict(self, X):
        return self.sigmoid(X @ self.W + self.b)
            
    def gradient_descent(self, X, Y, lr, lambda_):
        Y_pred = self.predict(X)
        
        dY = 
        

    def fit(self, X, Y, n_iteration=10000, lr=0.01, lambda_=0, print_accuracy=True, check_every=1000):
        self.check_shape(X, Y)
        
        for i in range(n_iteration):
            self.gradient_descent(X, Y)
            
            if print_accuracy and i % max(1, check_every) == 0:
                pass
            
        
        
            
        
