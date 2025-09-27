import pandas as pd

df = pd.read_csv("test1.csv")
df_clean = df.dropna()
corr_matrix = df_clean.corr()
pd.options.display.float_format = "{:.16f}".format
print(corr_matrix)
corr_matrix.to_csv("testout.csv", index=False)
