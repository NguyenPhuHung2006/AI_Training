import pandas as pd
import numpy as np
# def preprocessing_data(file_path, has_label=True):
#     target_cols = ["ID", "code", "Label"] if has_label else ["ID", "code"]
    
#     # 1. Load data with initial type hints to save memory
#     df = pd.read_csv(file_path, usecols=target_cols, low_memory=False)
    
#     # 2. Clean 'code' column (must not be empty)
#     df["code"] = df["code"].fillna("").astype(str)
#     df = df[df["code"].str.strip() != ""]
    
#     # 3. Handle 'ID' column (force to numeric, drop non-numeric)
#     df["ID"] = pd.to_numeric(df["ID"], errors='coerce')
#     df = df.dropna(subset=["ID"])
#     df["ID"] = df["ID"].astype(int)
    
#     if has_label:
#         # 4. Handle 'Label' column
#         # errors='coerce' turns strings/garbage into NaN
#         df["Label"] = pd.to_numeric(df["Label"], errors='coerce')
        
#         # 5. Drop rows with missing labels
#         df = df.dropna(subset=["Label"])
        
#         # 6. Strictly keep only binary 0 and 1
#         # This prevents the negative loss issue
#         df = df[df["Label"].isin([0, 1])]
#         df["Label"] = df["Label"].astype(int)
        
#     return df

def check_auto_increment_behavior(df, column="ID"):
    # 1. Sort by ID to ensure sequence
    ids = df[column].sort_values().values
    
    # 2. Check if the difference between every consecutive ID is exactly 1
    diffs = np.diff(ids)
    is_incremental = np.all(diffs == 1)
    
    # 3. Check for uniqueness
    is_unique = df[column].is_unique
    
    if is_incremental and is_unique:
        print(f"Column '{column}' behaves like an auto-incremented sequence.")
    else:
        print(f"Column '{column}' is NOT strictly incremental.")
        if not is_unique:
            print("Reason: Contains duplicate IDs.")
        else:
            print(f"Reason: Contains gaps or is out of order. Max diff: {diffs.max()}, Min diff: {diffs.min()}")
df_train = pd.read_csv("data/submission (3).csv")
check_auto_increment_behavior(df_train)
print(df_train["Label"].unique())