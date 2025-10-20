import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import norm
import test_lib

path = "../data/quiz2.csv"
lam = 0.94

cov = test_lib.get_ew_cov(path, lam, has_date=True, is_rate=False)
print(cov)


mean = 0
sd = np.sqrt(cov[0, 0])


VaR = norm.ppf(0.05, loc=mean, scale=sd)
print("VaR:\n", abs(VaR))


data_path = "../data/quiz2.csv"

dt1 = pd.read_csv(data_path)
dt2 = dt1.copy().iloc[1:, 1:]
dt2.index = range(dt2.shape[0])
dt2 = (dt2 - dt1.iloc[:-1, 1:]) / dt1.iloc[:-1, 1:]

x_centered = dt2["B"] - np.mean(dt2["B"])

nu, mu, sigma = t.fit(x_centered)

var95 = t.ppf(0.05, nu, loc=mu, scale=sigma)

print("VaR:\n", abs(var95))
