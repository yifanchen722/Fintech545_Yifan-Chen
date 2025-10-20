import pandas as pd
import numpy as np
from scipy.stats import t

df = np.loadtxt("/Users/nico/Desktop/input/test7_2.csv", delimiter=",", skiprows=1)

nu, mu, sigma = t.fit(df)

var95 = t.ppf(0.05, nu, loc=mu, scale=sigma)
mean_diff = abs(mu - var95)

print("VaR:\n", abs(var95))
print("Mean - VaR:\n", mean_diff)
