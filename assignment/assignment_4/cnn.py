import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from ai_model.deep_learning.nn_torch import CNN
from ai_model.deep_learning.nn_torch.callback import EarlyStopping, ModelCheckpoint

def url_to_matrix(url, size):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(size)
        return np.array(img).astype('float32') / 255.0
    except Exception as e:
        return np.zeros((size[0], size[1], 3), dtype='float32')

def read_data(train_path, test_path, img_size=(64, 64)):
    df_train = pd.read_csv(train_path, encoding='latin1')
    df_test = pd.read_csv(test_path, encoding='latin1')
    
    print("Downloading and processing training images...")
    X_train = np.array([url_to_matrix(url, img_size) for url in df_train['url_thumbnail']])
    
    print("Downloading and processing test images...")
    X_test = np.array([url_to_matrix(url, img_size) for url in df_test['url_thumbnail']])
    
    print(X_train.shape)

    le = LabelEncoder()
    y_train = le.fit_transform(df_train['label'])
    
    return X_train, y_train, X_test, len(le.classes_)
    

def main():
    X_train, y_train, X_test = read_data("data/train.csv", "data/test.csv")

if __name__ == "__main__":
    main()
    print("completed")