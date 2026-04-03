import pandas as pd
import re
from vncorenlp import VnCoreNLP
from tqdm import tqdm
from io import StringIO

# ===== INIT VnCoreNLP =====
rdrsegmenter = VnCoreNLP(
    r"C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/VnCoreNLP-1.2/VnCoreNLP-1.1.1.jar",
    annotators="wseg",
    max_heap_size='-Xmx2g'
)

# ===== CLEAN RAW TEXT (AFTER encoding is fixed) =====
def clean_raw_text(text):
    text = str(text)

    # remove emojis / special symbols (keep Vietnamese)
    text = re.sub(r'[^\w\sÀ-ỹà-ỹ]', ' ', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ===== TOKEN CLEANING =====
def clean_tokens(tokens):
    cleaned = []

    for token in tokens:
        token = token.lower()

        # remove punctuation-only tokens
        if re.fullmatch(r"[^\w\s]+", token):
            continue

        # replace numbers
        if re.fullmatch(r"\d+(\.\d+)?", token):
            cleaned.append("<num>")
            continue

        # split underscore words
        if "_" in token:
            parts = token.split("_")
            for p in parts:
                if len(p) > 1:
                    cleaned.append(p)
        else:
            if len(token) > 1:
                cleaned.append(token)

    return cleaned


# ===== MAIN TEXT PROCESSING =====
def process_text(text):
    text = clean_raw_text(text)

    sentences = rdrsegmenter.tokenize(text)
    tokens = [t for sent in sentences for t in sent]

    tokens = clean_tokens(tokens)

    return " ".join(tokens)


# ===== READ + FIX ENCODING (CRITICAL PART) =====
def read_and_fix_csv(input_path):
    # read raw bytes
    with open(input_path, "rb") as f:
        raw = f.read()

    # try utf-8 first
    try:
        text = raw.decode("utf-8")
    except:
        # fallback if needed
        text = raw.decode("latin1", errors="ignore")

    # load into pandas
    df = pd.read_csv(StringIO(text))

    return df


# ===== MAIN PIPELINE =====
def process_csv(input_path, output_path):
    df = read_and_fix_csv(input_path)

    assert "product_name" in df.columns, "Missing product_name column"

    tqdm.pandas()

    df["product_name"] = df["product_name"].progress_apply(process_text)

    # save clean UTF-8 file
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved to {output_path}")


# ===== RUN =====
if __name__ == "__main__":
    input_csv = "C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/data/test.csv"
    output_csv = "C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/test.csv"

    process_csv(input_csv, output_csv)