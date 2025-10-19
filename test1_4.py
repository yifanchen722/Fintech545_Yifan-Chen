import pandas as pd

df = pd.read_csv("../data/test1.csv")
pairwise_corr = df.corr()

print(pairwise_corr)
