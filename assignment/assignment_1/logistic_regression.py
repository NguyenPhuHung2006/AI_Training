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

    
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def gradient_descent(X_train, y_train, w, b, lr, lambda_=0):
    y_predict = sigmoid(X_train @ w + b)
    
    m = X_train.shape[0]
    
    dw = (X_train.T @ (y_predict - y_train) + lambda_ * w) / m
    db = np.mean(y_predict - y_train)
    
    w -= lr * dw
    b -= lr * db
    
    return w, b

def compute_cost(X, y, w, b, lambda_):
    py = sigmoid(X @ w + b)
    eps = 1e-8
    cost = -np.mean(y * np.log(py + eps) + (1 - y) * np.log(1 - py + eps))
    reg = (lambda_ * np.sum(w ** 2)) / (2 * X.shape[0]) 
    
    return cost + reg

def compute_accuracy(X, y, w, b):
    y_predict = predict(X, w, b)
    y_predict = y_predict.reshape(-1)
    y = y.reshape(-1)
    return np.mean(y_predict == y, axis=0)
    
def predict(X_test, w, b):
    return sigmoid(X_test @ w + b) >= 0.5

def get_validation(X_train, y_train, info):
    lr = info["lr"]
    num_iterations = info["num_iterations"]
    lambda_ = info["lambda_"]
    w = info["w"]
    b = info["b"]
    check_interval = info["check_interval"]
    print_cost = info["print_cost"]

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    for i in range(0, num_iterations):
        w, b = gradient_descent(X_tr, y_tr, w, b, lr, lambda_)
        if i % check_interval == 0:
            if print_cost:
                cost_train = compute_cost(X_tr, y_tr, w, b, lambda_)
                cost_val = compute_cost(X_val, y_val, w, b, lambda_)
                print(f"cost :{i} | train: {cost_train:.4f} | validation: {cost_val:.4f}")
            else:
                accuracy_train = compute_accuracy(X_tr, y_tr, w, b)
                accuracy_val = compute_accuracy(X_val, y_val, w, b)
                print(f"accuracy :{i} | train: {accuracy_train * 100:.4f}% | validation: {accuracy_val * 100:.4f}%")
    

def get_result(X_train, y_train, X_test, passenger_id, info):
    lr = info["lr"]
    num_iterations = info["num_iterations"]
    lambda_ = info["lambda_"]
    w = info["w"]
    b = info["b"]
    check_interval = info["check_interval"]
    
    for i in range(0, num_iterations):
        w, b = gradient_descent(X_train, y_train, w, b, lr, lambda_)
        if i % check_interval == 0:
            cost = compute_cost(X_train, y_train, w, b, lambda_)
            accuracy = compute_accuracy(X_train, y_train, w, b)
            print(f"i:{i} | cost: {cost:.4f} | accuracy: {accuracy * 100:.4f}%")
            
    y_test = predict(X_test, w, b)
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
    
    normalized = False

    X_train_df, stats = fit_preprocess(X_train_raw, normalized)
    X_test_df = apply_preprocess(X_test_raw, stats, normalized)
    
    X_train = X_train_df.to_numpy(dtype=float)
    X_test  = X_test_df.to_numpy(dtype=float) 
        
    info = {
        "lr": 0.02,
        "num_iterations": 10000,
        "lambda_": 0.02,
        "w": np.zeros((X_train.shape[1], 1)),
        "b": 0.0,
        "check_interval": 1000,
        "print_cost": False
    }
    
    # get_result(X_train, y_train, X_test, passenger_id, info)
    # get_validation(X_train, y_train, info)
    visualize_pca(X_train, y_train)

if __name__ == "__main__":
    main()
    print("completed")
