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

def add_total_spend(df):
    df = df.copy()
    df["TotalSpend"] = df[
        ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    ].sum(axis=1)
    
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    
    return df

def add_features(df):
    df = df.copy()
    df["SpendPerAge"] = df["TotalSpend"] / (df["Age"] + 1)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["LuxurySpend"] = df["Spa"] + df["VRDeck"]
    df["Cryo_NoSpend_Mismatch"] = (
        (df["CryoSleep"] == False) & (df["NoSpend"] == 1)
    ).astype(int)
    
    df["CabinNum_scaled"] = df["CabinNum"] / df["CabinNum"].max()
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,18,30,45,60,100], labels=False)
    return df

def add_group(df_train, df_test):
    df_train, df_test = df_train.copy(), df_test.copy()
    train_group = df_train["PassengerId"].str.split("_").str[0]
    test_group  = df_test["PassengerId"].str.split("_").str[0]

    group_size = train_group.value_counts()

    df_train["GroupSize"] = train_group.map(group_size)
    df_test["GroupSize"]  = test_group.map(group_size).fillna(1)
    
    return df_train, df_test

def read_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    passenger_id = df_test["PassengerId"]
    
    df_train, df_test = add_group(df_train, df_test)
    drop_cols = ["PassengerId", "Name"]
    df_train = df_train.drop(columns=drop_cols)
    df_test  = df_test.drop(columns=drop_cols)

    # split cabin column
    df_train = split_cabin_column(df_train)
    df_test = split_cabin_column(df_test)

    # numerical columns
    num_cols = [
        "RoomService", "FoodCourt", "ShoppingMall",
        "Spa", "VRDeck", "Age", "CabinNum"
    ]
    num_medians = df_train[num_cols].median()
    df_train[num_cols] = df_train[num_cols].fillna(num_medians)
    df_test[num_cols] = df_test[num_cols].fillna(num_medians)
    
    df_train = add_total_spend(df_train)
    df_test = add_total_spend(df_test)
    
    df_train = add_features(df_train)
    df_test = add_features(df_test)

    # categorical columns
    cat_cols = ["CryoSleep", "VIP", "HomePlanet", "Destination", "Deck", "Side"]
    
    y_train = df_train["Transported"].to_numpy().astype(int)
    df_train = df_train.drop(columns="Transported")

    full = pd.concat([df_train, df_test], axis=0)
    full = fillna_cat_cols(full, cat_cols)

    df_train = full.iloc[:len(y_train)]
    df_test  = full.iloc[len(y_train):]

    df_train = df_train.reindex(columns=df_test.columns, fill_value=0)

    x_train = df_train.to_numpy().astype(float)
    x_test = df_test.to_numpy().astype(float)

    return x_train, y_train, x_test, passenger_id

class GB_Base(ABC):
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def __init__(self, learning_rate: float, max_depth, min_child_weight, lambda_=0, gamma=0, n_trees=100):
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
        
        # columns subsampling
        colsample = 0.7
        features = np.random.choice(
            n_features,
            int(colsample * n_features),
            replace=False
        )

        for feature in features:
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
            
            # row subsampling
            subsample = 0.7
            idx = np.random.choice(
                len(y),
                int(subsample * len(y)),
                replace=False
            )

            X_sub = X[idx]
            g_sub = gradients[idx]
            h_sub = hessians[idx]

            tree = self.build_decision_tree(X_sub, g_sub, h_sub, self.lambda_)
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
            
            # row subsampling
            subsample = 0.7
            idx = np.random.choice(
                len(y_train),
                int(subsample * len(y_train)),
                replace=False
            )

            X_sub = X_train[idx]
            g_sub = gradients[idx]
            h_sub = hessians[idx]

            tree = self.build_decision_tree(X_sub, g_sub, h_sub, self.lambda_)

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
    
          
def get_validation(X_train, y_train, model):
    model.fit_with_validation(X_train, y_train)

def get_result(X_train, y_train, X_test, passenger_id, model):
    
    model.fit(X_train, y_train, print_every=10)
        
    y_predict = model.predict_label(X_test).astype(bool)

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
    
    model = GB_Classification(
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        lambda_=1,
        gamma=0.3,
        n_trees=400
    )

    get_result(X_train, y_train, X_test, passenger_id, model)
    # get_validation(X_train, y_train, model)

if __name__ == "__main__":
    main()
    print("completed")