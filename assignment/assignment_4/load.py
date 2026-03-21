import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm.contrib.concurrent import thread_map 

def url_to_matrix(url, size):
    """Downloads an image and converts it to a (H, W, 3) matrix."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=7) # Slightly longer timeout
        response.raise_for_status() 
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(size)
        # float16 is the 'Safety Net' for 20k samples at 224/256
        return np.array(img).astype('float16') / 255.0
    except Exception:
        # Crucial: Use the 'size' variable passed in to ensure matching shapes
        return np.zeros((size[1], size[0], 3), dtype='float16')

def download_and_save(train_path, test_path, img_size=(128, 128), save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_train = pd.read_csv(train_path, encoding='latin1')
    df_test = pd.read_csv(test_path, encoding='latin1')

    # 1. Train Set
    print(f"Downloading {len(df_train)} Training Images at {img_size}...")
    train_list = thread_map(
        lambda url: url_to_matrix(url, img_size), 
        df_train['url_thumbnail'], 
        max_workers=15, # Slightly slower to avoid being banned by Shopee
        desc="Train Progress"
    )
    X_train = np.stack(train_list)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    del train_list, X_train # Clear memory immediately

    # 2. Test Set
    print(f"\nDownloading {len(df_test)} Test Images...")
    test_list = thread_map(
        lambda url: url_to_matrix(url, img_size), 
        df_test['url_thumbnail'], 
        max_workers=15,
        desc="Test Progress"
    )
    X_test = np.stack(test_list)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    del test_list, X_test # Clear memory immediately

    # 3. Labels
    np.save(os.path.join(save_dir, "y_train.npy"), df_train['label'].values)

def main():
    # Recommended Master Size: 128 or 224
    # If using 256, ensure you use float16 as shown above
    download_and_save(
        train_path="data/train.csv", 
        test_path="data/test.csv",
        img_size=(224, 224) 
    )

if __name__ == "__main__":
    main()