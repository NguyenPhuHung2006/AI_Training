import pandas as pd
import torch
import os
from torch.nn.utils.rnn import pad_sequence
from ai_model.deep_learning.nn_torch import TextCNN
from ai_model.deep_learning.nn_torch.callback import EarlyStopping, ModelCheckpoint
from tokenizers import ByteLevelBPETokenizer
import numpy as np

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
    tokenizer.train(files=["temp_code.txt"], vocab_size=5000, min_frequency=2)
    os.remove("temp_code.txt")
    
    return tokenizer
    
def get_tokenize(df, tokenizer, MAX_T=1024, has_label=True, pad_id=None):
    encoded_codes = []
    for text in df["code"]:
        token_ids = tokenizer.encode(text).ids
        truncated_ids = token_ids[:MAX_T]
        encoded_codes.append(torch.tensor(truncated_ids, dtype=torch.long))

    pad_id = pad_id if pad_id is not None else 0
    padded_sequences = pad_sequence(encoded_codes, batch_first=True, padding_value=pad_id)
    
    labels_tensor = None
    if has_label:
        labels_tensor = torch.tensor(df["Label"].tolist(), dtype=torch.float).unsqueeze(1)
    
    return padded_sequences, labels_tensor

def main():
    df_train = preprocessing_data("data/train.csv")
    df_test = preprocessing_data("data/test.csv", has_label=False)
    
    # tokenizer = tokenizing(df_train)
    tokenizer = ByteLevelBPETokenizer(
        "nn_data/tokenizer/vocab.json",
        "nn_data/tokenizer/merges.txt"
    )
    pad_id = tokenizer.token_to_id("<pad>")
    
    MAX_T = 200
    X_train, y_train = get_tokenize(df_train, tokenizer, has_label=True, MAX_T=MAX_T, pad_id=pad_id)
    X_test, _ = get_tokenize(df_test, tokenizer, has_label=False, MAX_T=MAX_T, pad_id=pad_id)
    vocab_size = tokenizer.get_vocab_size()
    
    model = TextCNN(
        cost="bce",
        lr=3e-4,
        weight_decay=5e-4
    )

    model.pad_token = pad_id
    # model.add_embedding(vocab_size=vocab_size, embed_dim=256)
    embedding_matrix = torch.load("nn_data/embedding.pt", weights_only=False)
    model.set_embedding_matrix(embedding_matrix)
    model.embedding.weight.requires_grad = True

    # model.add_filter(out_channels=96, kernel_size=2, activation="relu", dropout=0.3)
    model.add_filter(out_channels=96, kernel_size=3, activation="relu", dropout=0.3)
    model.add_filter(out_channels=96, kernel_size=5, activation="relu", dropout=0.3)

    model.add_pool("max", pool_dropout=0.5)

    model.add_fc(n_units=128, activation="relu", dropout=0.5)
    model.add_fc(n_units=1)
    model.build()
    
    model.set_scheduler("reduce_on_plateau", mode="min", factor=0.5, patience=3)
    
    nn_data_path = "nn_data"
    os.makedirs(nn_data_path, exist_ok=True)
    
    i = 0
    while os.path.exists(f"{nn_data_path}/rnn_{i}.pth"):
        i += 1
        
    # reload the model
    # model.load(f"{nn_data_path}/rnn_{i - 1}.pth")

    callbacks = [EarlyStopping(patience=5), ModelCheckpoint(f"{nn_data_path}/rnn_{i}.pth")]
    
    print(f"Starting training with Vocab Size: {vocab_size} and Max Sequence: {X_train.shape[1]}")
    
    model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        callbacks=callbacks
    )
    
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    y_pred = y_pred.flatten()
    
    df = pd.DataFrame({
        "ID": np.arange(0, len(y_pred)),
        "Label": y_pred
    })
    
    output_path = "outputs/nn"
    os.makedirs(output_path, exist_ok=True)

    df.to_csv(f"{output_path}/rnn.csv", index=False)
    

if __name__ == "__main__":
    main()
    print("Completed!")