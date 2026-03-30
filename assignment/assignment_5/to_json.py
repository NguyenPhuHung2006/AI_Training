import pandas as pd
import re
import json

KEYWORDS = {
    "int","float","double","char","void","long","short","unsigned","signed",
    "if","else","switch","case","for","while","do","break","continue","return",
    "struct","class","public","private","protected","static","const","enum",
    "typedef","sizeof","union","goto","default","extern","register","volatile",
    "inline","bool","true","false","namespace","using","try","catch","throw",
    "new","delete","this","operator"
}

SPECIAL_TOKENS = {"VAR", "TYPE", "CONST", "FUNC", "NUM", "STR", "CHAR"}


# -------------------------
# Normalize identifiers
# -------------------------
def normalize_identifiers(code: str) -> str:
    def repl(match):
        word = match.group(0)

        if word in KEYWORDS or word in SPECIAL_TOKENS:
            return word
        elif word.isupper():
            return "CONST"
        elif word[0].isupper():
            return "TYPE"
        else:
            return "VAR"

    return re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', repl, code)


# -------------------------
# Detect function calls safely
# -------------------------
def replace_func(match):
    word = match.group(0)
    if word in KEYWORDS:
        return word
    return "FUNC"


# -------------------------
# Safe truncation
# -------------------------
def safe_truncate(code: str, max_len=1000) -> str:
    if len(code) <= max_len:
        return code

    truncated = code[:max_len]

    # cut to last full line
    if '\n' in truncated:
        truncated = truncated[:truncated.rfind('\n')]

    return truncated


# -------------------------
# Main cleaning function
# -------------------------
def clean_code(code: str) -> str:
    # 1. Remove comments
    code = re.sub(r'/\*.*?(\*/|$)', ' ', code, flags=re.DOTALL)
    code = re.sub(r'//.*?$', ' ', code, flags=re.MULTILINE)

    # 2. Remove non-printable characters
    code = re.sub(r'[^\x20-\x7E\n]', ' ', code)

    # 3. Normalize strings
    code = re.sub(r'"(\\.|[^"\\])*"', 'STR', code)
    code = re.sub(r"'(\\.|[^'\\])*'", 'CHAR', code)

    # 4. Normalize numbers
    code = re.sub(r'\b\d+\b', 'NUM', code)

    # 5. Detect function calls
    code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b(?=\s*\()', replace_func, code)

    # 6. Normalize identifiers
    code = normalize_identifiers(code)

    # 7. Operators
    code = re.sub(r'([=!<>]=|&&|\|\|)', r' \1 ', code)
    code = re.sub(r'([{}()\[\];,=+\-*/<>])', r' \1 ', code)

    # 8. Normalize whitespace
    code = re.sub(r'\n+', '\n', code)
    code = re.sub(r'[ \t\r\f\v]+', ' ', code)
    code = re.sub(r' *\n *', '\n', code)

    # remove excessive newlines
    code = re.sub(r'\n{2,}', '\n', code)

    # 9. Trim and safely truncate
    code = code.strip()
    code = safe_truncate(code)

    return code


# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/train.csv", usecols=["ID", "code", "Label"]).fillna('')

# Apply cleaning
df['clean_code'] = df['code'].apply(clean_code)

# Optional: drop empty/broken samples
df = df[df['clean_code'].str.len() > 0]

# -------------------------
# Export to JSONL
# -------------------------
with open("data/clean_code.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        json.dump({
            "ID": str(row['ID']),
            "code": row['clean_code'],
            "Label": str(row['Label'])
        }, f, ensure_ascii=False)
        f.write("\n")

print("Export completed: data/clean_code.jsonl")