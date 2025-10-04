import pandas as pd
import numpy as np
from scipy.stats import norm

data = np.loadtxt("/Users/nico/Desktop/input/test7_1.csv", delimiter=",", skiprows=1)

mean = np.mean(data, axis=0)
sd = np.std(data, axis=0, ddof=1)

VaR = norm.ppf(0.05, loc=mean, scale=sd)
mean_diff = abs(mean - VaR)

print("VaR:\n", abs(VaR))
print("Mean - VaR:\n", mean_diff)

output_path = "/Users/nico/Desktop/output_8.1.csv"
pd.DataFrame({"VaR": [abs(VaR)], "Diff from Mean": [mean_diff]}).to_csv(
    output_path, index=False, header=True
)
