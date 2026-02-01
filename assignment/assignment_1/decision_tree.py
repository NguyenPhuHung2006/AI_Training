import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_cabin_column(df):
    df = df.copy()
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0].fillna("Unknown")
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2].fillna("Unknown")
    df = df.drop(columns=["Cabin"])
    return df
    
def fillna_cat_column(df, cat_cols):
    df = df.copy()
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def read_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    passenger_id = df_test["PassengerId"]
    
    # drop unnecessary column
    drop_cols = ["PassengerId", "Name"]
    df_train = df_train.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols)
    
    # split the cabin column
    df_train = split_cabin_column(df_train)
    df_test = split_cabin_column(df_test)
    
    # fillna of categorical columns
    cat_cols = ["CryoSleep", "VIP", "HomePlanet", "Destination", "Deck", "Side"]
    df_train = fillna_cat_column(df_train, cat_cols)
    df_test = fillna_cat_column(df_test, cat_cols)
    
    # fillna of numerical columns
    num_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Age", "CabinNum"]
    num_medians = df_train[num_cols].median()
    df_train[num_cols] = df_train[num_cols].fillna(num_medians)
    df_test[num_cols] = df_test[num_cols].fillna(num_medians)
    
    y_train = df_train["Transported"].to_numpy().astype(int)
    df_train = df_train.drop(columns="Transported")
    df_train = df_train.reindex(columns=df_test.columns, fill_value=0)
    
    x_train = df_train.to_numpy().astype(float)
    x_test = df_test.to_numpy().astype(float)
    
    return x_train, y_train, x_test, passenger_id
        
def compute_entropy(p):
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def compute_score(x, y):
    thresholds = np.unique(x)
    n_thresholds = len(thresholds)
    n_samples = len(y)
    best_compute_score, best_compute_threshold = float("inf"), None
        
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
        
        score = w_left * compute_entropy(p_left) + w_right * compute_entropy(p_right)
        if score < best_compute_score:
            best_compute_score = score
            best_compute_threshold = threshold
            
    return best_compute_score, best_compute_threshold
    
def best_split(X, y):
    best_feature, best_threshold = None, None
    best_score = float("inf")
    
    n_features = X.shape[1]
    
    for feature in range(n_features):
        score, threshold = compute_score(X[:, feature], y)
        if score < best_score:
            best_score = score
            best_feature = feature
            best_threshold = threshold
    
    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
def most_frequent_value(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)].astype(bool)
    
def build_decision_tree(X, y, depth=0, max_depth=5, min_samples=10):
    if (max_depth and depth >= max_depth) or len(y) < min_samples or len(np.unique(y)) == 1:
        return Node(value=most_frequent_value(y))
    
    feature, threshold = best_split(X, y)
    
    # in case there's a bug in the code
    if feature is None or threshold is None:
        return Node(value=most_frequent_value(y))
    
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    left = build_decision_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples)
    right = build_decision_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples)
    
    return Node(feature=feature, threshold=threshold, left=left, right=right)

def predict_tree(x, node):
    if node.value is not None:
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_tree(x, node.left)
    return predict_tree(x, node.right)

def predict_tree_batch(X, root):
    return np.array([predict_tree(x, root) for x in X])

def get_validation(X_train, y_train, info):
    max_depth = info["max_depth"]
    min_samples = info["min_samples"]
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    root = build_decision_tree(X_tr, y_tr, max_depth=max_depth, min_samples=min_samples)
    y_train_predict = predict_tree_batch(X_tr, root)
    y_val_predict = predict_tree_batch(X_val, root)
    
    print(f"accuracy | train:{np.mean(y_train_predict == y_tr)} | validation: {np.mean(y_val_predict == y_val)}")

def get_result(X_train, y_train, X_test, passenger_id, info):
    max_depth = info["max_depth"]
    min_samples = info["min_samples"]
    root = build_decision_tree(X_train, y_train, max_depth=max_depth, min_samples=min_samples)
    y_predict = predict_tree_batch(X_test, root)
    
    df_out = pd.DataFrame({
        "PassengerId": passenger_id,
        "Transported": y_predict 
    })
    
    df_out.to_csv("outputs/decision_tree.csv", index=False)

def main():
    X_train, y_train, X_test, passenger_id = read_data("data/train.csv", "data/test.csv")
    
    info = {
        "max_depth": 8,
        "min_samples": 20,
    }

    get_result(X_train, y_train, X_test, passenger_id, info)
    # get_validation(X_train, y_train, info)
    
if __name__ == "__main__":
    main()
    print("completed")