import pandas as pd

def filter_data(df, has_label=True):
    target_cols = ["ID", "code", "Label"] if has_label else ["ID", "code"]    
    numeric_cols = ["ID", "Label"] if has_label else ["ID"]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["code"] = df["code"].astype(str).replace(['nan', 'None', ''], pd.NA)
    df = df.dropna(subset=target_cols)

    if has_label:
        df["Label"] = df["Label"].astype(int)
    
    return df

def fix_data(df):
    return df

df_train = pd.read_csv("data/train.csv", low_memory=False)
df_test = pd.read_csv("data/test.csv", low_memory=False)

df_train = fix_data(df_train)
df_test = fix_data(df_test)