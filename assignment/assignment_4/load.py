import numpy as np
import pandas as pd
import os
import time
import requests
from io import BytesIO
from PIL import Image
from tqdm.contrib.concurrent import thread_map
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------
# 1. INTERNET CHECK
# ---------------------------
def wait_for_internet(interval=5):
    while True:
        try:
            requests.get("https://www.google.com", timeout=5)
            return
        except requests.RequestException:
            print("[NO INTERNET] Waiting...")
            time.sleep(interval)

# ---------------------------
# 2. SESSION WITH RETRY
# ---------------------------
def create_session():
    session = requests.Session()

    retries = Retry(
        total=2,  # keep low to avoid retry explosion
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

session = create_session()

# ---------------------------
# 3. DOWNLOAD IMAGE
# ---------------------------
def url_to_matrix(url, size, retries=2):
    for attempt in range(retries):
        wait_for_internet()

        try:
            # small delay to reduce network stress
            time.sleep(0.1)

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = session.get(url, headers=headers, timeout=10)

            # skip bad URLs
            if response.status_code == 404:
                return None

            response.raise_for_status()

            img = Image.open(BytesIO(response.content))

            # FIX: normalize all formats safely
            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize(size)

            return np.array(img, dtype=np.float16) / 255.0

        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)

        except Exception:
            return None

# ---------------------------
# 4. PROCESS DATAFRAME
# ---------------------------
def process_dataframe(df, img_size, max_workers):
    urls = df["url_thumbnail"].tolist()

    results = thread_map(
        lambda url: url_to_matrix(url, img_size),
        urls,
        max_workers=max_workers,  # safer for unstable internet
        desc="Downloading"
    )

    images = []
    failed_urls = []

    for url, img in zip(urls, results):
        if img is None:
            failed_urls.append(url)
            images.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.float16))
        else:
            images.append(img)

    X = np.stack(images)
    return X, failed_urls

# ---------------------------
# 5. MAIN PIPELINE
# ---------------------------
def build_dataset(train_csv, test_csv, img_size=(128, 128), max_workers=3):
    os.makedirs("data", exist_ok=True)

    # ===== TRAIN =====
    print("Loading train data...")
    df_train = pd.read_csv(train_csv)

    X_train, failed_train = process_dataframe(df_train, img_size, max_workers)

    # handle labels (auto encode if needed)
    if df_train["label"].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train = le.fit_transform(df_train["label"])
        np.save("data/label_encoder_classes.npy", le.classes_)
    else:
        y_train = df_train["label"].values

    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)

    print(f"✓ Saved X_train, y_train")
    print(f"⚠ Failed train images: {len(failed_train)}")

    if failed_train:
        pd.Series(failed_train).to_csv("data/failed_train_urls.csv", index=False)

    # ===== TEST =====
    print("\nLoading test data...")
    df_test = pd.read_csv(test_csv)

    X_test, failed_test = process_dataframe(df_test, img_size, max_workers)
    test_ids = df_test["ID"].values

    np.save("data/X_test.npy", X_test)
    np.save("data/test_ids.npy", test_ids)

    print(f"✓ Saved X_test, test_ids")
    print(f"⚠ Failed test images: {len(failed_test)}")

    if failed_test:
        pd.Series(failed_test).to_csv("data/failed_test_urls.csv", index=False)

# ---------------------------
# 6. RUN
# ---------------------------
if __name__ == "__main__":
    build_dataset(
        train_csv="data/train.csv",
        test_csv="data/test.csv",
        img_size=(128, 128),
        max_workers=3  # IMPORTANT: keep low for stability
    )