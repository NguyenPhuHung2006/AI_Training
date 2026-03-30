import pandas as pd

# Load CSV
df = pd.read_csv(
    "data/train.csv",
    usecols=["ID", "code", "Label"],  # only the 3 columns you need
)

# Replace NaN with an empty string (or something else)
df = df.fillna('')

# Use a safe separator
sep = '\x1f'

# Combine columns as strings
df['combined'] = df['ID'].astype(str) + sep + df['code'].astype(str) + sep + df['Label'].astype(str)

# Write to TXT
with open("data/combined.txt", "w", encoding="utf-8") as f:
    for line in df['combined']:
        f.write(str(line) + "\n")  # ensure it's a string


print("completed")