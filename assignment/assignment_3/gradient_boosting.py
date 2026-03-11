import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

def process_dates(df):
    df = df.copy()
    
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%y %H:%M", errors="coerce")
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format="%d-%m-%y", errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format="%d-%m-%y", errors="coerce")
    
    df = df.assign(
        search_month = df["date_time"].dt.month,
        stay_days = (df["srch_co"] - df["srch_ci"]).dt.days.clip(lower=0),
        booking_lead = (df["srch_ci"] - df["date_time"]).dt.days.clip(lower=0),
        search_weekday = df["date_time"].dt.weekday
    )

    df = df.drop(["date_time", "srch_ci", "srch_co"], axis=1)

    return df

def add_missing(df):
    df = df.copy()
    
    df["dest_missing"] = df["d1"].isna().astype(int)
    df["distance_missing"] = df["orig_destination_distance"].isna().astype(int)
    
    return df
    
def read_data(train_path, test_path, dest_path):
    df_train = pd.read_csv(train_path, nrows=500000)
    # df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_dest = pd.read_csv(dest_path)
        
    df_train = df_train.merge(df_dest, on="srch_destination_id", how="left")
    df_test = df_test.merge(df_dest, on="srch_destination_id", how="left")
    
    df_train = process_dates(df_train)
    df_test = process_dates(df_test)
    
    df_train = add_missing(df_train)
    df_test = add_missing(df_test)

    y_train = df_train["hotel_cluster"]

    df_train = df_train.drop(columns=["hotel_cluster", "user_id"])
    df_test = df_test.drop(columns=["id", "user_id"])
    
    df_test = df_test.reindex(columns=df_train.columns)
    
    X_train = df_train.astype("float32").to_numpy()
    y_train = y_train.to_numpy()
    X_test = df_test.astype("float32").to_numpy()
        
    return X_train, y_train, X_test
    
            
def main():
    X_train, y_train, X_test = read_data("new_dataset/train.csv", "new_dataset/test.csv", "new_dataset/destinations.csv")
        
    print("data preprocessing completed")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.05, random_state=42
    )
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    params = {
        "objective": "multi:softprob",
        "num_class": 100,
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "merror",
        "nthread": 4
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50
    )
    
    y_test = model.predict(dtest)

    top5 = np.argsort(-y_test, axis=1)[:, :5]
    labels = np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, top5)
        
    df = pd.DataFrame({
        "id": np.arange(0, len(labels)),
        "hotel_cluster": labels
    })

    df.to_csv("outputs/xgboost.csv", index=False)
    
        
if __name__ == '__main__':
    main()
    print("completed")
