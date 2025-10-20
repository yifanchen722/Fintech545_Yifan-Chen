import pandas as pd
import numpy as np
from scipy.stats import t

df = np.loadtxt("../data/test7_2.csv", delimiter=",", skiprows=1)

nu, mu, sigma = t.fit(df)

K = 100000

samples = np.random.uniform(low=0, high=1, size=K)
for i in range(K):
    samples[i] = t.ppf(samples[i], nu, loc=mu, scale=sigma)

# empirical distribution
unique_sorted = np.sort(np.unique(samples))
arr = np.empty(len(unique_sorted))
for i in range(len(unique_sorted)):
    arr[i] = np.sum(samples == unique_sorted[i]) / len(samples)
for i in range(1, len(unique_sorted)):
    arr[i] = arr[i - 1] + arr[i]

# calculate VAR and ES
var95 = 0
for i in range(len(arr)):
    if arr[i] >= 0.05:
        var95 = unique_sorted[i - 1]
        break

mean_diff = abs(mu - var95)
print("VAR=", abs(var95))
print("mean_diff=", mean_diff)
