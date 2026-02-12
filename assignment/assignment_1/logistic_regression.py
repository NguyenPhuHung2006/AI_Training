import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
        
def get_validation(X_train, y_train, info, model: LogisticRegression):
    lr = info["lr"]
    num_iterations = info["num_iterations"]
    lambda_ = info["lambda_"]
    check_interval = info["check_interval"]
    print_accuracy = info["print_accuracy"]
    
    model.fit_with_validation(X_train, y_train, lr=lr, n_iteration=num_iterations, lambda_=lambda_, 
                              print_accuracy=print_accuracy, check_every=check_interval)
    

def get_result(X_train, y_train, X_test, passenger_id, info, model: LogisticRegression):
    lr = info["lr"]
    num_iterations = info["num_iterations"]
    lambda_ = info["lambda_"]
    check_interval = info["check_interval"]
    
    model.fit(X_train, y_train, n_iteration=num_iterations, lr=lr, lambda_=lambda_, check_every=check_interval)
            
    y_test = model.predict(X_test)
    y_test = y_test.reshape(-1)
    
    df_out = pd.DataFrame({
        "PassengerId": passenger_id,
        "Transported": y_test
    })

    df_out.to_csv("outputs/logistic_regression.csv", index=False)
    
def visualize_pca(X, y):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure()
    plt.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=y.reshape(-1),
    )
    plt.title("PCA visualization (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Transported")
    plt.show()

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    
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
        
    info = {
        "lr": 0.02,
        "num_iterations": 10000,
        "lambda_": 0.02,
        "check_interval": 1000,
        "print_accuracy": True
    }
    
    model = LogisticRegression(X_train.shape[1])
    
    # get_result(X_train, y_train, X_test, passenger_id, info, model)
    get_validation(X_train, y_train, info, model)
    # visualize_pca(X_train, y_train)

if __name__ == "__main__":
    main()
    print("completed")
