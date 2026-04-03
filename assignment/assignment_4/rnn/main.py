import pandas as pd
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from ai_model.deep_learning.nn_torch import TextCNN
from ai_model.deep_learning.nn_torch.callback import EarlyStopping, ModelCheckpoint
import numpy as np
import json

with open("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/nn_data/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

pad_id = vocab["<pad>"]

def preprocessing_data(file_path, has_label=True, label_mapping=None):
    df = pd.read_csv(file_path)

    if has_label:
        df = df[["ID", "product_name", "label"]]
    else:
        df = df[["ID", "product_name"]]

    df["product_name"] = df["product_name"].fillna("").astype(str)
    df = df[df["product_name"].str.strip() != ""].copy()

    if has_label:
        df["label"] = df["label"].astype(str).str.strip()
        
        if label_mapping is None:
            unique_labels = sorted(df["label"].unique())
            label_mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        
        df["label"] = df["label"].map(label_mapping)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    return df, label_mapping

def text_to_ids(text, vocab):
    tokens = str(text).lower().split()
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

def get_tokenize(df, vocab, MAX_T=100, has_label=True):
    encoded_texts = []
    actual_indices = []

    for idx, row in df.iterrows():
        token_ids = text_to_ids(row["product_name"], vocab)
        token_ids = token_ids[:MAX_T]

        if len(token_ids) == 0:
            continue

        encoded_texts.append(torch.tensor(token_ids, dtype=torch.long))
        actual_indices.append(idx)

    if not encoded_texts:
        raise ValueError("No valid sequences found.")

    padded = pad_sequence(encoded_texts, batch_first=True, padding_value=pad_id)

    labels = None
    if has_label:
        labels = torch.tensor(df.loc[actual_indices, "label"].values, dtype=torch.long)

    return padded, labels, actual_indices

def main():
    df_train, label_mapping = preprocessing_data("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/train.csv")
    df_test, _ = preprocessing_data("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/test.csv", has_label=False)

    MAX_T = 40
    X_train, y_train, _ = get_tokenize(df_train, vocab, MAX_T=MAX_T, has_label=True)
    X_test, _, test_indices = get_tokenize(df_test, vocab, MAX_T=MAX_T, has_label=False)
    
    n_classes = len(label_mapping)
    vocab_size = len(vocab)

    model = TextCNN(
        cost="cce",
        lr=1e-3,
        weight_decay=5e-4,
        pad_id=pad_id
    )

    embedding_matrix = torch.load("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/nn_data/embedding.pt", weights_only=False)
    model.set_embedding_matrix(embedding_matrix)
    model.embedding.weight.requires_grad = True

    model.add_filter(out_channels=64, kernel_size=2, activation="relu", dropout=0.3)
    model.add_filter(out_channels=64, kernel_size=3, activation="relu", dropout=0.3)
    model.add_filter(out_channels=64, kernel_size=5, activation="relu", dropout=0.3)
    model.add_pool("max", pool_dropout=0.5)
    model.add_fc(n_units=128, activation="relu", dropout=0.5)
    model.add_fc(n_units=n_classes)
    model.build()

    model.set_scheduler("reduce_on_plateau", mode="min", factor=0.5, patience=2)

    nn_data_path = "C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/nn_data"
    os.makedirs(nn_data_path, exist_ok=True)

    i = 0
    while os.path.exists(f"{nn_data_path}/text_cnn_{i}.pth"):
        i += 1

    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(f"{nn_data_path}/text_cnn_{i}.pth")
    ]

    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=callbacks)

    y_pred_probs = model.predict(X_test)
    y_pred_ids = np.argmax(y_pred_probs, axis=1)

    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    final_labels = [inv_label_mapping[idx] for idx in y_pred_ids]

    df_output = pd.DataFrame({
        "ID": df_test.loc[test_indices, "ID"].values,
        "Label": final_labels
    })

    output_path = "C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/outputs"
    os.makedirs(output_path, exist_ok=True)
    df_output.to_csv(f"{output_path}/text_cnn.csv", index=False)

if __name__ == "__main__":
    main()