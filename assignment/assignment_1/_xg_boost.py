import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

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
    
          
def get_validation(X_train, y_train, model):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

def main():
    X_train, y_train, X_test, passenger_id = read_data(
        "data/train.csv",
        "data/test.csv"
    )
    
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=8,
        early_stopping_rounds=50
    )
    
    get_validation(X_train, y_train, model)

if __name__ == "__main__":
    main()
    print("completed")