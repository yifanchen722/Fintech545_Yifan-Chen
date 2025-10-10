import pandas as pd
import numpy as np
from scipy.stats import t

df = np.loadtxt("../data/test7_2.csv", delimiter=",", skiprows=1)

nu, mu, sigma = t.fit(df)

K = 100000
samples = t.rvs(df=nu, loc=mu, scale=sigma, size=K)

samples_sorted = np.sort(samples)

alpha = 0.05

var95_index = int(np.floor(alpha * K))
var95 = samples_sorted[var95_index]

ES = samples_sorted[:var95_index].mean()

es_diff = abs(mu - ES)

print("ES Absolute:\n", abs(ES))
print("ES Diff from Mean:\n", es_diff)
