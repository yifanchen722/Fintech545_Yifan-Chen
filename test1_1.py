import pandas as pd

df = pd.read_csv("test1.csv")
df_clean = df.dropna()
cov_matrix = df_clean.cov()
pd.options.display.float_format = "{:.16f}".format
print(cov_matrix)
cov_matrix.to_csv("testout.csv", index=False)
