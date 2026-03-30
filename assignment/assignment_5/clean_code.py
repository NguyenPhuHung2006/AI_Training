import pandas as pd
import re
import json

# Load CSV
df = pd.read_csv("data/train.csv", usecols=["ID", "code", "Label"]).fillna('')

# Function to remove comments and whitespace
def clean_code(code: str, max_space=10) -> str:
    # Remove /* */ block comments (even if unterminated)
    code = re.sub(r'/\*.*?(\*/|$)', '', code, flags=re.DOTALL)
    # Remove // line comments
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    # Remove whitespace sequences longer than max_space
    # Example: any \s repeated more than max_space
    code = re.sub(r'\s{' + str(max_space) + r',}', '', code)
    return code

# Apply cleaning
df['clean_code'] = df['code'].apply(clean_code)

# Export to JSON Lines
with open("data/clean_code.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json.dump({
            "ID": str(row['ID']),
            "code": row['clean_code'],
            "Label": str(row['Label'])
        }, f, ensure_ascii=False)
        f.write("\n")

print("Export completed: data/clean_code.jsonl")