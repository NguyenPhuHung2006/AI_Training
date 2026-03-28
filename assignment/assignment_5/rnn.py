import pandas as pd
import torch
import json
import re
import os
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from ai_model.deep_learning.nn_torch import RNN
from ai_model.deep_learning.nn_torch.callback import EarlyStopping, ModelCheckpoint

def tokenize(code):
    if pd.isna(code):
        code = ""
    return re.findall(r"\w+|[^\s\w]", str(code))

def encode(tokens, vocab):
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=4, ensure_ascii=False)

def load_vocab(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    df = pd.read_csv(
        "data/train.csv",
        usecols=["code", "Label"],
        dtype={"code": str},
        low_memory=False
    )
    df["Label"] = pd.to_numeric(df["Label"], errors='coerce')
    df = df.dropna(subset=["Label", "code"])
    df["Label"] = df["Label"].astype(int)

    codes = df["code"].tolist()
    labels = df["Label"].tolist()

    tokenized_codes = [tokenize(c) for c in codes]

    counter = Counter()
    for tokens in tokenized_codes:
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for word in counter:
        vocab[word] = len(vocab)

    encoded_codes = [torch.tensor(encode(tokens, vocab), dtype=torch.long) for tokens in tokenized_codes]

    padded_sequences = pad_sequence(encoded_codes, batch_first=True, padding_value=vocab["<pad>"])
    labels_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)

    model = RNN(mode="many_to_one", cost="bce")
    model.add_embedding(vocab_size=len(vocab), embed_dim=128)
    model.add_rnn(hidden_size=128, bidirectional=True)
    model.add_fc(1)
    model.build()
    
    nn_data_path = "nn_data"
    os.makedirs(nn_data_path, exist_ok=True)
    
    i = 0
    while os.path.exists(f"{nn_data_path}/rnn_{i}.pth"):
        i += 1
        
    # reload the model
    # model.load(f"{nn_data_path}/rnn_{i - 1}.pth")

    callbacks = [EarlyStopping(patience=3), ModelCheckpoint(f"{nn_data_path}/rnn_{i}.pth")]
    model.fit(
        padded_sequences, 
        labels_tensor, 
        epochs=10, 
        batch_size=32, 
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
    print("Completed!")