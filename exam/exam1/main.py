import librosa
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import ast
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
            
    def predict_prob(self, x):
        probs = [tree.predict(x) for tree in self.forest]
        return np.mean(probs)
    
    def predict_batch_prob(self, X):
        return np.array([self.predict_prob(x) for x in X])
            
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

class GB_Base(ABC):
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, default_left=True):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.default_left = default_left
    
    def __init__(self, 
                 learning_rate: float, 
                 max_depth, 
                 min_child_weight, 
                 min_samples_leaf, 
                 sub_sample_size=0.8,
                 sub_feature_size=0.8,
                 lambda_=0, 
                 gamma=0, 
                 alpha=0, 
                 n_trees=100
                ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.sub_sample_size = sub_sample_size
        self.sub_feature_size = sub_feature_size
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
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
    def leaf_value(self, gradients, hessians):
        pass
    
    @abstractmethod
    def evaluate(self, y, y_pred):
        pass
    
    def soft_threshold(self, G):
        if G > self.alpha:
            return G - self.alpha
        elif G < -self.alpha:
            return G + self.alpha
        return 0
    
    def gain(self, G_left, H_left, G_right, H_right, G_total, H_total, eps=1e-15):

        GL = self.soft_threshold(G_left)
        GR = self.soft_threshold(G_right)
        GP = self.soft_threshold(G_total)

        left_score = (GL ** 2) / (H_left + self.lambda_ + eps)
        right_score = (GR ** 2) / (H_right + self.lambda_ + eps)
        parent_score = (GP ** 2) / (H_total + self.lambda_ + eps)

        return 0.5 * (left_score + right_score - parent_score) - self.gamma
    
    def compute_score(self, x, gradients, hessians):
        
        nan_mask = np.isnan(x)
        not_nan_mask = ~nan_mask
        
        x_valid = x[not_nan_mask]
        g_valid = gradients[not_nan_mask]
        h_valid = hessians[not_nan_mask]

        g_nan = gradients[nan_mask]
        h_nan = hessians[nan_mask]

        G_nan = np.sum(g_nan)
        H_nan = np.sum(h_nan)

        if len(x_valid) == 0:
            return -float("inf"), None, True
        
        sorted_idx = np.argsort(x_valid)
        x_sorted = x_valid[sorted_idx]
        g_sorted = g_valid[sorted_idx]
        h_sorted = h_valid[sorted_idx]

        G_total = np.sum(g_sorted) + G_nan
        H_total = np.sum(h_sorted) + H_nan
        
        G_valid_total = np.sum(g_sorted)
        H_valid_total = np.sum(h_sorted)

        G_left = 0.0
        H_left = 0.0

        best_gain = -float("inf")
        best_threshold = None
        best_default_left = True
        
        for i in range(len(x_sorted) - 1):

            G_left += g_sorted[i]
            H_left += h_sorted[i]
            
            if x_sorted[i] == x_sorted[i + 1]:
                continue

            G_right = G_valid_total - G_left
            H_right = H_valid_total - H_left
            
            if H_left < self.min_child_weight or H_right < self.min_child_weight:
                continue                    
            
            gain_left = self.gain(
                G_left + G_nan,
                H_left + H_nan,
                G_right,
                H_right,
                G_total,
                H_total
            )
            
            gain_right = self.gain(
                G_left,
                H_left,
                G_right + G_nan,
                H_right + H_nan,
                G_total,
                H_total
            )

            if gain_left > best_gain:
                best_gain = gain_left
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                best_default_left = True

            if gain_right > best_gain:
                best_gain = gain_right
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                best_default_left = False
                
        return best_gain, best_threshold, best_default_left
    
    def get_random_indices(self, ratio, total_size, replace=False):
        size = max(1, int(ratio * total_size))
        return np.random.choice(total_size, size, replace=replace)
    
    def best_split(self, X, gradients, hessians):
        best_feature = None
        best_threshold = None
        best_gain = -float("inf")
        best_default_left = True

        n_features = X.shape[1]
        
        features = self.get_random_indices(self.sub_feature_size, n_features, replace=False)

        for feature in features:
            gain, threshold, default_left = self.compute_score(
                X[:, feature], gradients, hessians
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                best_default_left = default_left

        return best_feature, best_threshold, best_gain, best_default_left
    
    def build_decision_tree(self, X, gradients, hessians, depth=0):
        H = np.sum(hessians)
        if depth >= self.max_depth or H < self.min_child_weight:
            return self.Node(value=self.leaf_value(gradients, hessians))
        
        feature, threshold, gain, default_left = self.best_split(X, gradients, hessians)
        
        # in case there's a bug in the code
        if feature is None or threshold is None or gain <= 0:
            return self.Node(value=self.leaf_value(gradients, hessians))
        
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        nan_mask = np.isnan(X[:, feature])

        if default_left:
            left_mask = left_mask | nan_mask
        else:
            right_mask = right_mask | nan_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self.Node(value=self.leaf_value(gradients, hessians))
        
        left = self.build_decision_tree(
            X[left_mask], 
            gradients[left_mask], 
            hessians[left_mask], 
            depth + 1
        )
        
        right = self.build_decision_tree(
            X[right_mask], 
            gradients[right_mask], 
            hessians[right_mask], 
            depth + 1
        )
        
        return self.Node(
            feature=feature, 
            threshold=threshold, 
            left=left, 
            right=right, 
            default_left=default_left
        )
    
    def _predict_tree(self, x, node: Node):
        while node.value is None:
            value = x[node.feature]

            if np.isnan(value):
                if node.default_left:
                    node = node.left
                else:
                    node = node.right

            elif value <= node.threshold:
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
            
            idx = self.get_random_indices(self.sub_sample_size, len(y), replace=False)

            X_sub = X[idx]
            g_sub = gradients[idx]
            h_sub = hessians[idx]

            tree = self.build_decision_tree(X_sub, g_sub, h_sub)
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
            
            idx = self.get_random_indices(self.sub_sample_size, len(y_train), replace=False)

            X_sub = X_train[idx]
            g_sub = gradients[idx]
            h_sub = hessians[idx]

            tree = self.build_decision_tree(X_sub, g_sub, h_sub)

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

    def leaf_value(self, gradients, hessians):
        H = np.sum(hessians)
        G = np.sum(gradients)
        return - super().soft_threshold(G) / (H + self.lambda_)
    
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

    def leaf_value(self, gradients, hessians):
        H = np.sum(hessians)
        G = np.sum(gradients)
        return - super().soft_threshold(G) / (H + self.lambda_)
    
    def predict_label(self, X):
        probs = self.predict_prob(X)
        return (probs >= 0.5).astype(int)
    
    def predict_prob(self, X):
        y_pred = super().predict(X)
        return self.sigmoid(y_pred)
    
    def evaluate(self, y, y_pred):
        p = self.sigmoid(y_pred)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# ----------------------------
# Feature extraction
# ----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std])


# ----------------------------
# Label encoding (FIXED)
# ----------------------------
def encode_label_string(label_string, n_classes=19):
    binary = np.zeros(n_classes)

    if pd.isna(label_string):
        return binary

    labels = ast.literal_eval(label_string)

    for idx in labels:
        binary[int(idx)] = 1

    return binary

def convert_predictions_to_strings(y_pred_bin):
    result = []

    for row in y_pred_bin:
        indices = np.where(row == 1)[0].tolist()

        if len(indices) == 0:
            result.append("")  
        else:
            result.append(str(indices))

    return result

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    df_all = pd.read_csv("all_data.csv")
    df_labels = pd.read_csv("train_labels.csv")

    df_train = pd.merge(df_all, df_labels, on="rec_id", how="inner")
    df_test = df_all[~df_all["rec_id"].isin(df_labels["rec_id"])]
    
    print("df_train shape:", df_train.shape)
    print("df_test shape:", df_test.shape)
    
    X_train = []
    y_train = []
    X_test = []
    
    for row in df_train.itertuples():
        try:
            features = extract_features("src_wavs/" + row.filename + ".wav")
            label_vector = encode_label_string(row.labels)

            X_train.append(features)
            y_train.append(label_vector)

        except Exception as e:
            print(f"Error processing {row.filename}: {e}")

    for row in df_test.itertuples():
        try:
            features = extract_features("src_wavs/" + row.filename + ".wav")
            X_test.append(features)

        except Exception as e:
            print(f"Error processing {row.filename}: {e}")

    X_train = np.array(X_train)
    y_train = np.vstack(y_train)
    X_test = np.array(X_test)

    models_xg = []
    models_rf = []
    for k in range(19):
        model_xg = GB_Classification(
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=1,
            min_samples_leaf=5,
            n_trees=100
        )
        
        model_rf = RandomForest(max_depth=None, min_samples=10, n_trees=100)

        model_xg.fit(X_train, y_train[:, k])
        model_rf.fit(X_train, y_train[:, k])
        models_xg.append(model_xg)
        models_rf.append(model_rf)
        
        print(f"training model {k + 1}")

    y_pred = np.column_stack([
        model.predict_prob(X_test)
        for model in models_xg
    ])
    
    y_pred = np.column_stack([
        0.65 * models_xg[i].predict_prob(X_test) + 0.35 * models_rf[i].predict_batch_prob(X_test)
        for i in range(len(models_xg))
    ])
    
    result = []
    for i in range(X_test.shape[0]):
        for j in range(y_pred.shape[1]):
            result.append([df_test.iloc[i, 0] * 100 + j, y_pred[i][j]])
            
    result = np.array(result)
            
    df_submission = pd.DataFrame(columns=["Id", "Probability"])
    df_submission["Id"] = result[:, 0]
    df_submission["Probability"] = result[:, 1]

    df_submission.to_csv("submission.csv", index=False)

    print("Submission saved.")


if __name__ == "__main__":
    main()
    print("completed")