import pandas as pd
# df = pd.read_csv("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/train.csv")

# lengths = df["product_name"].apply(lambda x: len(str(x).split()))
# print("Max:", lengths.max())
# print("Mean:", lengths.mean())
# print("90%:", lengths.quantile(0.90))
# print("95%:", lengths.quantile(0.95))
# print("99%:", lengths.quantile(0.99))

df = pd.read_csv("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/test.csv")
print(df["product_name"].head())

# import pandas as pd

# with open("C:/STUDY/HK2_2/LAP_TRINH_4/ise/assignment/assignment_4/rnn/data/test.csv", "rb") as f:
#     raw = f.read()
    
# fixed = raw.decode("utf-8", errors="ignore")
# from io import StringIO

# df = pd.read_csv(StringIO(fixed))
# print(df["product_name"].head())
