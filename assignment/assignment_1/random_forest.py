import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def split_cabin_column(df):
    df = df.copy()
    cabin = df["Cabin"].str.split("/", expand=True)

    df["Deck"] = cabin[0].fillna("Unknown")
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2].fillna("Unknown")

    df = df.drop(columns=["Cabin"])
    return df


def fillna_cat_cols(df, cat_cols):
    df = df.copy()
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def read_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    passenger_id = df_test["PassengerId"]

    # drop unnecessary columns
    drop_cols = ["PassengerId", "Name"]
    df_train = df_train.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols)

    # split cabin column
    df_train = split_cabin_column(df_train)
    df_test = split_cabin_column(df_test)
    
    # categorical columns
    cat_cols = ["CryoSleep", "VIP", "HomePlanet", "Destination", "Deck", "Side"]
    df_train = fillna_cat_cols(df_train, cat_cols)
    df_test = fillna_cat_cols(df_test, cat_cols)

    # numerical columns
    num_cols = [
        "RoomService", "FoodCourt", "ShoppingMall",
        "Spa", "VRDeck", "Age", "CabinNum"
    ]
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
    thresholds = np.percentile(x, np.linspace(2, 98, 50))
    thresholds = np.unique(thresholds)
    n_samples = len(y)

    best_score = float("inf")
    best_threshold = None

    for threshold in thresholds:
        
        left = y[x <= threshold]
        right = y[x > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        w_left = len(left) / n_samples
        w_right = len(right) / n_samples

        score = (
            w_left * compute_entropy(left.mean())
            + w_right * compute_entropy(right.mean())
        )

        if score < best_score:
            best_score = score
            best_threshold = threshold

    return best_score, best_threshold


def best_split(X, y, feature_indices):
    best_feature = None
    best_threshold = None
    best_score = float("inf")

    for feature in feature_indices:
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
    return values[np.argmax(counts)]


def build_decision_tree(X, y, depth=0, max_depth=5, min_samples=10):
    if (
        (max_depth is not None and depth >= max_depth)
        or len(y) < min_samples
        or len(np.unique(y)) == 1
    ):
        return Node(value=most_frequent_value(y))

    n_features = X.shape[1]
    feature_indices = random_features_indices(n_features)

    feature, threshold = best_split(X, y, feature_indices)

    if feature is None or threshold is None:
        return Node(value=most_frequent_value(y))

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    left = build_decision_tree(
        X[left_mask], y[left_mask],
        depth + 1, max_depth, min_samples
    )
    right = build_decision_tree(
        X[right_mask], y[right_mask],
        depth + 1, max_depth, min_samples
    )

    return Node(feature=feature, threshold=threshold, left=left, right=right)


def predict_tree(x, node):
    if node.value is not None:
        return node.value

    if x[node.feature] <= node.threshold:
        return predict_tree(x, node.left)

    return predict_tree(x, node.right)


def predict_tree_batch(X, root):
    return np.array([predict_tree(x, root) for x in X])


def random_sample(X, y):
    n = len(X)
    idx = np.random.choice(n, size=n, replace=True)
    return X[idx], y[idx]


def random_features_indices(n_features):
    m = int(np.sqrt(n_features))
    return np.random.choice(n_features, m, replace=False)


def build_random_forest(X, y, info):
    forest = []

    for i in range(info["n_trees"]):
        
        if i % 5 == 0:
            print(i)
        
        Xb, yb = random_sample(X, y)

        tree = build_decision_tree(
            Xb,
            yb,
            max_depth=info["max_depth"],
            min_samples=info["min_samples"]
        )

        forest.append(tree)

    return forest


def predict_forest(x, forest):
    votes = [predict_tree(x, tree) for tree in forest]
    return Counter(votes).most_common(1)[0][0].astype(bool)


def predict_forest_batch(X, forest):
    return np.array([predict_forest(x, forest) for x in X])


def get_validation(X_train, y_train, info):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    forest = build_random_forest(X_tr, y_tr, info)

    train_acc = np.mean(predict_forest_batch(X_tr, forest) == y_tr)
    val_acc = np.mean(predict_forest_batch(X_val, forest) == y_val)

    print(f"accuracy | train: {train_acc} | validation: {val_acc}")


def get_result(X_train, y_train, X_test, passenger_id, info):
    forest = build_random_forest(X_train, y_train, info)

    y_predict = predict_forest_batch(X_test, forest)

    df_out = pd.DataFrame({
        "PassengerId": passenger_id,
        "Transported": y_predict
    })

    df_out.to_csv("outputs/random_forest.csv", index=False)
    
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
    X_train, y_train, X_test, passenger_id = read_data(
        "data/train.csv",
        "data/test.csv"
    )

    info = {
        "max_depth": None,
        "min_samples": 10,
        "n_trees": 100
    }

    # get_result(X_train, y_train, X_test, passenger_id, info)
    # get_validation(X_train, y_train, info)
    visualize_pca(X_train, y_train)

if __name__ == "__main__":
    main()
    print("completed")
