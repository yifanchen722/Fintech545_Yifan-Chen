import pandas as pd

df = pd.read_csv("../data/problem5.csv")
pairwise_cov = df.cov()

print(pairwise_cov)
