import pandas as pd

df = pd.read_csv("../data/test1.csv")
pairwise_cov = df.cov()

print(pairwise_cov)
pairwise_cov.to_csv("../output/testout.csv", index=False)
