import pandas as pd

df = pd.read_csv("test1.csv")
pairwise_cov = df.cov()
pd.options.display.float_format = "{:.16f}".format
print(pairwise_cov)
pairwise_cov.to_csv("testout.csv", index=False)
