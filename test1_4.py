import pandas as pd

df = pd.read_csv("test1.csv")
pairwise_corr = df.corr()
pd.options.display.float_format = "{:.16f}".format
print(pairwise_corr)
pairwise_corr.to_csv("testout.csv", index=False)
