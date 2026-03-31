from gensim.models import Word2Vec
import pandas as pd
import torch
import os
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import random
import torch.nn.functional as F
random.seed(42)
np.random.seed(42)

def preprocessing_data(file_path, has_label=True):
    target_cols = ["ID", "code", "Label"] if has_label else ["ID", "code"]
    
    df = pd.read_csv(file_path, usecols=target_cols, dtype={"code": str}, low_memory=False)
    
    numeric_cols = ["ID", "Label"] if has_label else ["ID"]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["code"] = df["code"].astype(str).replace(['nan', 'None', ''], pd.NA)
    df = df.dropna(subset=target_cols)

    if has_label:
        df["Label"] = df["Label"].astype(int)
    
    return df
        
def tokenizing(df):
    with open("temp_code.txt", "w", encoding="utf-8") as f:
        for code_snippet in df["code"]:
            f.write(code_snippet + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.add_special_tokens([
        "<pad>",
        "<sos>",
        "<eos>"
    ])
    tokenizer.train(files=["temp_code.txt"], vocab_size=10000, min_frequency=2)
    os.remove("temp_code.txt")
    
    return tokenizer
    
def build_w2v_sentences(df, tokenizer):
    sentences = []
    
    for text in df["code"]:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        sentences.append(tokens)
    
    return sentences

def train_word2vec(sentences, embed_dim):
    w2v = Word2Vec(
        sentences,
        vector_size=embed_dim,
        window=3,
        min_count=1,
        workers=4,
        sg=1,
        seed=42
    )
    return w2v

def build_embedding_matrix(tokenizer, w2v, embed_dim):
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    embedding_matrix = np.random.randn(vocab_size, embed_dim) * 0.1

    for token, idx in vocab.items():
        if token in w2v.wv:
            embedding_matrix[idx] = w2v.wv[token]
            
    pad_id = tokenizer.token_to_id("<pad>")
    embedding_matrix[pad_id] = np.zeros(embed_dim)

    return torch.tensor(embedding_matrix, dtype=torch.float32)

# main
df_train = preprocessing_data("data/train_clean_code.csv")
df_test = preprocessing_data("data/test_clean_code.csv", has_label=False)
df_code = pd.concat([
    df_train[["code"]],
    df_test[["code"]]
], axis=0, ignore_index=True)

tokenizer = tokenizing(df_code)

# Build Word2Vec training data
sentences = build_w2v_sentences(df_code, tokenizer)

# Train Word2Vec
w2v = train_word2vec(sentences, embed_dim=256)

# Build embedding matrix
embedding_matrix = build_embedding_matrix(tokenizer, w2v, embed_dim=256)
embedding_matrix = F.normalize(embedding_matrix, p=2, dim=1)

torch.save(embedding_matrix, "nn_data/embedding.pt")
tokenizer.save_model("nn_data/tokenizer/")
print("completed")