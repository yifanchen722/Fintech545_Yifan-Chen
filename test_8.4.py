import pandas as pd
import numpy as np
from scipy.stats import norm

data = np.loadtxt("../data/test7_1.csv", delimiter=",", skiprows=1)

mean = np.mean(data, axis=0)
sd = np.std(data, axis=0, ddof=1)

alpha = 0.05
VaR = norm.ppf(alpha, loc=mean, scale=sd)

ES = mean - sd * norm.pdf(norm.ppf(alpha)) / alpha
es_diff = abs(mean - ES)

print("ES Absolute:\n", abs(ES))
print("ES Diff from Mean:\n", es_diff)
