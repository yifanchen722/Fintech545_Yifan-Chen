import pandas as pd
import numpy as np
from scipy.stats import t
import test_lib

data_path = "../data/problem6.csv"

dt1 = pd.read_csv(data_path)
dt2 = dt1.copy().iloc[1:, :-1]
dt2.index = range(dt2.shape[0])
dt2 = (dt2 - dt1.iloc[:-1, :-1]) / dt1.iloc[:-1, :-1]

x_centered = dt2["x3"] - np.mean(dt2["x3"])

nu, mu, sigma = t.fit(x_centered)

print("Fitted parameters (nu, mu, sigma):", nu, mu, sigma)

var95 = t.ppf(0.05, nu, loc=mu, scale=sigma)
alpha = 0.05

t_quantile = t.ppf(alpha, df=nu)
ES = mu - (sigma * (nu + t_quantile**2) / (nu - 1) * t.pdf(t_quantile, df=nu) / alpha)

es_diff = abs(mu - ES)
print("ES:\n", abs(ES))

print("VaR:\n", abs(var95))
