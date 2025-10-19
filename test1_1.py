import pandas as pd

df = pd.read_csv("../data/test1.csv")
df_clean = df.dropna()
print(df_clean)

cov_matrix = df_clean.cov()
print(cov_matrix)
