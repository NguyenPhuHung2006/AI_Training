from gensim.models import Word2Vec
import pandas as pd
import torch
import numpy as np
import random
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)

# ===== BUILD SENTENCES (NO TOKENIZER) =====
def build_sentences(df):
    sentences = []
    
    for text in df["product_name"]:
        tokens = str(text).split()   # 🔥 simple and correct
        sentences.append(tokens)
    
    return sentences


# ===== TRAIN WORD2VEC =====
def train_word2vec(sentences, embed_dim):
    w2v = Word2Vec(
        sentences,
        vector_size=embed_dim,
        window=3,
        min_count=2,
        workers=4,
        sg=1,
        seed=42
    )
    return w2v


# ===== BUILD VOCAB =====
def build_vocab(w2v):
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    idx = 2
    for word in w2v.wv.index_to_key:
        vocab[word] = idx
        idx += 1

    return vocab


# ===== BUILD EMBEDDING MATRIX =====
def build_embedding_matrix(vocab, w2v, embed_dim):
    vocab_size = len(vocab)
    
    embedding_matrix = np.random.randn(vocab_size, embed_dim) * 0.1

    for token, idx in vocab.items():
        if token in w2v.wv:
            embedding_matrix[idx] = w2v.wv[token]

    # pad = zero vector
    embedding_matrix[vocab["<pad>"]] = np.zeros(embed_dim)

    all_vecs = np.array([w2v.wv[word] for word in w2v.wv.index_to_key])
    embedding_matrix[vocab["<unk>"]] = all_vecs.mean(axis=0)

    return torch.tensor(embedding_matrix, dtype=torch.float32)


# ===== MAIN =====
df_train = pd.read_csv("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/train.csv")
df_test = pd.read_csv("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/test.csv")

df_all = pd.concat([
    df_train[["product_name"]],
    df_test[["product_name"]]
], axis=0, ignore_index=True)

# ===== BUILD DATA =====
sentences = build_sentences(df_all)

# ===== TRAIN W2V =====
embed_dim = 100
w2v = train_word2vec(sentences, embed_dim)

# ===== BUILD VOCAB =====
vocab = build_vocab(w2v)

# ===== BUILD EMBEDDING =====
embedding_matrix = build_embedding_matrix(vocab, w2v, embed_dim)
embedding_matrix = F.normalize(embedding_matrix, p=2, dim=1)

# ===== SAVE =====
torch.save(embedding_matrix, "C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/nn_data/embedding.pt")

# save vocab
import json
with open("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/nn_data/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("completed")