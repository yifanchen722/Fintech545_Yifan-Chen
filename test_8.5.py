import pandas as pd
import numpy as np
from scipy.stats import t

df = np.loadtxt("../data/test7_2.csv", delimiter=",", skiprows=1)

nu, mu, sigma = t.fit(df)

alpha = 0.05

t_quantile = t.ppf(alpha, df=nu)
ES = mu - (sigma * (nu + t_quantile**2) / (nu - 1) * t.pdf(t_quantile, df=nu) / alpha)

es_diff = abs(mu - ES)
print("ES:\n", abs(ES))
print("Diff from Mean:\n", es_diff)
