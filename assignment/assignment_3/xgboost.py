import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
    
def fill_na_median(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = df[col].fillna(df[col].median())
            
    return df

def process_dates(df):
    df = df.copy()
    
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], errors="coerce")

    df = df.assign(
        search_month = df["date_time"].dt.month,
        stay_days = (df["srch_co"] - df["srch_ci"]).dt.days.clip(lower=0),
        booking_lead = (df["srch_ci"] - df["date_time"]).dt.days.clip(lower=0)
    )

    df = df.drop(["date_time", "srch_ci", "srch_co"], axis=1)

    return df

def normalize(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()
    binary_cols = ["is_mobile", "is_package"]
    
    for col in df_train.columns:
        if df_train[col].dtype != "object" and col not in binary_cols:
            mean = df_train[col].mean()
            std = df_train[col].std()
            df_train[col] = (df_train[col] - mean) / (std + 1e-8)
            df_test[col] = (df_test[col] - mean) / (std + 1e-8)
            
    return df_train, df_test
    
def read_data(train_path, test_path, dest_path):
    df_train = pd.read_csv(train_path, nrows=200000)
    df_test = pd.read_csv(test_path)
    df_dest = pd.read_csv(dest_path)
    
    df_train = df_train.merge(df_dest, on="srch_destination_id", how="left")
    df_test = df_test.merge(df_dest, on="srch_destination_id", how="left")
    
    df_train = process_dates(df_train)
    df_test = process_dates(df_test)

    y_train = df_train["hotel_cluster"]

    drop_cols = ["hotel_cluster", "is_booking", "cnt", "user_id"]

    df_train = df_train.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols, errors="ignore")
    df_test = df_test.drop(columns="id")
    
    df_train["orig_destination_distance"] = np.log1p(df_train["orig_destination_distance"])
    df_test["orig_destination_distance"] = np.log1p(df_test["orig_destination_distance"])

    df_train = fill_na_median(df_train)
    df_test = fill_na_median(df_test)
    
    df_train, df_test = normalize(df_train, df_test)
    
    X_train = df_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = df_test.to_numpy()
        
    return X_train, y_train, X_test
    
            
def main():
    X_train, y_train, X_test = read_data("new_dataset/train.csv", "new_dataset/test.csv", "new_dataset/destinations.csv")
        
    print("data preprocessing completed")
    
    model = XGBClassifier(
        tree_method="hist",
        objective="multi:softprob",
        num_class=100,
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="merror"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=True
    )
    
    y_test = model.predict_proba(X_test)

    top5 = np.argsort(-y_test, axis=1)[:, :5]
    labels = np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, top5)
        
    df = pd.DataFrame({
        "id": np.arange(0, len(labels)),
        "labels": labels
    })

    df.to_csv("outputs/xgboost.csv", index=False)
    
        
if __name__ == '__main__':
    main()
    print("completed")
