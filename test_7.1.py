import pandas as pd

file_path = "../data/test7_1.csv"
col_name = "x1"

df = pd.read_csv(file_path)
mu = df[col_name].mean()
sigma = df[col_name].std(ddof=1)

print("mu:", mu)
print("sigma:", sigma)
