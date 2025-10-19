import pandas as pd

df = pd.read_csv("../data/test1.csv")
df_clean = df.dropna()
corr_matrix = df_clean.corr()

print(corr_matrix)
