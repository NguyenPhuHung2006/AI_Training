import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd

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
    def leaf_value(self, gradients, hessians, lambda_):
        pass
    
    @abstractmethod
    def evaluate(self, y, y_pred):
        pass
    
class GB_MSE(GB_BaseLoss):
    
    def init_prediction(self, y):
        return np.mean(y)
    
    def gradient(self, y, y_pred):
        return y_pred - y   

    def hessian(self, y, y_pred):
        return np.ones_like(y)

    def leaf_value(self, gradients, hessians, lambda_):
        return - np.sum(gradients) / (np.sum(hessians) + lambda_ + 1e-15)
    
    def evaluate(self, y, y_pred):
        return 0.5 * np.mean((y - y_pred)**2)
    
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

    def leaf_value(self, gradients, hessians, lambda_):
        return - np.sum(gradients) / (np.sum(hessians) + lambda_ + 1e-15)
    
    def predict_label(self, y_pred):
        proba = self.sigmoid(y_pred)
        return (proba >= 0.5).astype(int)
    
    def evaluate(self, y, y_pred):
        preds = self.predict_label(y_pred)
        y = y.astype(int)
        return np.mean(preds == y)
        
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
        if depth >= self.max_depth or X.shape[0] < self.min_samples:
            return GB_Node(value=loss.leaf_value(gradients, hessians, lambda_))
        
        feature, threshold, gain = self.best_split(X, gradients, hessians, lambda_)
        
        # in case there's a bug in the code
        if feature is None or threshold is None or gain <= 0:
            return GB_Node(value=loss.leaf_value(gradients, hessians, lambda_))
        
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
    
    def fit(self, X, gradients, hessians, loss, lambda_):
        self.root = self.build_decision_tree(X, gradients, hessians, loss, lambda_)
                
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
    def __init__(self, loss, learning_rate: float, max_depth, min_samples, lambda_=0, n_trees=100):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.lambda_ = lambda_
        self.n_trees = n_trees
        self.trees = []
        self.init_pred = None
        
    def fit(self, X, y, check_every=None):
        self.trees = []
        self.init_pred = self.loss.init_prediction(y)
        y_pred = np.full_like(y, self.init_pred, dtype=float)
        
        for i in range(self.n_trees):
            gradients = self.loss.gradient(y, y_pred)
            hessians = self.loss.hessian(y, y_pred)

            tree = GB_DecisionTree(self.max_depth, self.min_samples)
            tree.fit(X, gradients, hessians, self.loss, self.lambda_)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)
            
            if i > 0 and check_every is not None and i % max(1, check_every) == 0:
                print(f"{i:03d} / {self.n_trees:04d}")
            
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_pred, dtype=float)
        
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
                
    def fit_with_validation(self, X, y, test_size=0.2, random_state=42, check_every=None):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        self.fit(X_train, y_train, check_every=check_every)
        
        y_train_predict = self.predict(X_train)
        y_val_predict = self.predict(X_val)
        
        print(
            f"train:{self.loss.evaluate(y_train, y_train_predict):.4f} | "
            f"validation:{self.loss.evaluate(y_val, y_val_predict):.4f}"
        )   
          
def get_validation(X_train, y_train, model):
    model.fit_with_validation(X_train, y_train, check_every=2)

def get_result(X_train, y_train, X_test, passenger_id, model):
    
    model.fit(X_train, y_train, 2)
    
    y_predict = model.predict(X_test)
    
    y_predict = model.loss.predict_label(y_predict).astype(bool)

    df_out = pd.DataFrame({
        "PassengerId": passenger_id,
        "Transported": y_predict
    })

    df_out.to_csv("outputs/gradient_boosting.csv", index=False)

def main():
    X_train, y_train, X_test, passenger_id = read_data(
        "data/train.csv",
        "data/test.csv"
    )
    
    model = GradientBoosting(GB_BCE(), 
                             learning_rate=0.05, 
                             max_depth=3, 
                             min_samples=10, 
                             lambda_=0, 
                             n_trees=200)

    get_result(X_train, y_train, X_test, passenger_id, model)
    # get_validation(X_train, y_train, model)

if __name__ == "__main__":
    main()
    print("completed")