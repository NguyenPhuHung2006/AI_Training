import numpy as np
import pandas as pd
import os
from ai_model.deep_learning.nn_torch import MLP
from ai_model.deep_learning.nn_torch.callback import EarlyStopping, ModelCheckpoint

def process_dates(df):
    df = df.copy(deep=False)
    
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

def fill_na_median(df_train, df_test):
    df_train = df_train.copy(deep=False)
    df_test = df_test.copy(deep=False)
    
    for col in df_train.columns:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            median = df_train[col].median()
            df_train[col] = df_train[col].fillna(median)
            df_test[col] = df_test[col].fillna(median)
            
    return df_train, df_test

def get_binary_cols(df):
    binary_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0,1}):
                binary_cols.append(col)

    return binary_cols

def log_normalize(df, cols):
    df = df.copy(deep=False)

    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
            df[col] = np.log1p(np.clip(df[col], a_min=0, a_max=None))

    return df

def cluster_probability_features(df_train, df_test, group_col, prefix):
    
    df = df_train.copy(deep=False)

    df["weight"] = df["is_booking"] * 4 + 1

    counts = (
        df.pivot_table(
            index=group_col,
            columns="hotel_cluster",
            values="weight",
            aggfunc="sum",
            fill_value=0
        )
    )

    probs = counts.div(counts.sum(axis=1), axis=0)

    probs.columns = [f"{prefix}_cluster_{c}" for c in probs.columns]

    df_train = df_train.merge(probs, on=group_col, how="left")
    df_test = df_test.merge(probs, on=group_col, how="left")

    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    return df_train, df_test

def standardization(df_train, df_test, cols):
    df_train = df_train.copy(deep=False)
    df_test = df_test.copy(deep=False)

    for col in cols:
        if pd.api.types.is_numeric_dtype(df_train[col]):

            mean = df_train[col].mean()
            std = df_train[col].std()

            if std == 0:
                std = 1.0

            df_train[col] = (df_train[col] - mean) / std
            df_test[col] = (df_test[col] - mean) / std

    return df_train, df_test
    
def read_data(train_path, test_path, dest_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_dest = pd.read_csv(dest_path)
    
    df_train = df_train.merge(df_dest, on="srch_destination_id", how="left")
    df_test = df_test.merge(df_dest, on="srch_destination_id", how="left")
    
    df_train = process_dates(df_train)
    df_test = process_dates(df_test)

    y_train = df_train["hotel_cluster"]
    cluster_features = [
        ("srch_destination_id", "dest"),
        ("hotel_market", "market"),
        ("hotel_country", "country"),
        ("user_location_country", "user_country"),
        ("user_location_city", "user_city")
    ]
    for col, prefix in cluster_features:
        df_train, df_test = cluster_probability_features(
            df_train, df_test, col, prefix
        )
    
    bookings = df_train[df_train["is_booking"] == 1]
    top_clusters = bookings["hotel_cluster"].value_counts().index[:5].tolist()

    df_train = df_train.drop(columns=["hotel_cluster", "user_id"])
    df_test = df_test.drop(columns=["id", "user_id"])
    
    # log normalization
    log_cols = ["orig_destination_distance", "cnt"]
    df_train = log_normalize(df_train, log_cols)
    df_test = log_normalize(df_test, log_cols)
    
    # fill missing
    df_train, df_test = fill_na_median(df_train, df_test)
    
    # standardization all except the binary cols
    binary_cols = get_binary_cols(df_train)
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    embed_cols_name = [
        "user_location_city",
        "user_location_region",
        "srch_destination_id",
        "hotel_market",
        "hotel_country"
    ]
    std_cols = [c for c in numeric_cols if c not in binary_cols and c not in embed_cols_name]
    
    df_train, df_test = standardization(df_train, df_test, std_cols)
    
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
    
    embed_cols = [
        (df_train.columns.get_loc(c), int(max(df_train[c].max(), df_test[c].max()) + 1))
        for c in embed_cols_name
    ]
        
    X_train = df_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = df_test.to_numpy()
        
    return X_train, y_train, X_test, top_clusters, embed_cols
            
def main():
    X_train, y_train, X_test, top_clusters, embed_cols = read_data("new_dataset/train.csv", 
                                                       "new_dataset/test.csv", 
                                                       "new_dataset/destinations.csv")
        
    print("data preprocessing completed")
    
    embed_indices = [idx for idx, _ in embed_cols]
    
    print("NaNs:", np.isnan(X_train).sum())
    print("Feature std mean:", 
        np.array([X_train[:, i] for i in range(X_train.shape[1]) if i not in embed_indices]).std(axis=0).mean()
    )
    print("Unique labels:", len(np.unique(y_train)))
    print("Feature count:", X_train.shape[1])

    model = MLP(
        cost="cce",
        lr=0.002,
        weight_decay=1e-4
    )
    
    for col_index, vocab_size in embed_cols:
        if vocab_size > 10000:
            d = 32
        elif vocab_size > 1000:
            d = 16
        else:
            d = 8
        model.add_embedding(col_index=col_index, vocab_size=vocab_size, embed_dim=d)
        
    num_cols = [i for i in range(X_train.shape[1]) if i not in embed_indices]
    
    model.set_numerical_cols(num_cols)
    
    model.add_layer(512, activation="relu", norm="batch", dropout=0.4, init="he")
    model.add_layer(256, activation="relu", norm="batch", dropout=0.3, init="he")
    model.add_layer(128, activation="relu", norm="batch", dropout=0.2, init="he")
    model.add_layer(100)

    model.build()
    
    nn_data_path = "nn_data"
    os.makedirs(nn_data_path, exist_ok=True)
    
    i = 0
    while os.path.exists(f"{nn_data_path}/nn_torch_{i}.pth"):
        i += 1
    
    # reload the model
    # model.load(f"{nn_data_path}/nn_torch_{i - 1}.pth")
        
    model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=4096,
        val_split=0.1,
        callbacks=[
            EarlyStopping(patience=10),
            ModelCheckpoint(f"{nn_data_path}/nn_torch_{i}.pth")
        ]
    )
        
    y_test = model.predict(X_test)
    top5 = np.argsort(-y_test, axis=1)[:, :5]
    total_replacement = 0
    for i in range(len(top5)):
        probs = y_test[i]
        sorted_probs = np.sort(probs)[::-1]
        if probs.max() < 0.2 or (sorted_probs[0] - sorted_probs[1]) < 0.05:
            top5[i] = top_clusters
            total_replacement += 1
    
    print(f"Total number of replacement is {total_replacement}")
            
    labels = np.apply_along_axis(lambda x: " ".join(map(str, x)), 1, top5)
        
    # save the result
    df = pd.DataFrame({
        "id": np.arange(0, len(labels)),
        "hotel_cluster": labels
    })
    
    output_path = "outputs/nn"
    os.makedirs(output_path, exist_ok=True)

    df.to_csv(f"{output_path}/nn_torch.csv", index=False)
        
if __name__ == '__main__':
    main()
    print("completed")