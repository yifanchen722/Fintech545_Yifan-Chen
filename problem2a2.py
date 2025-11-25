import pandas as pd
from scipy.stats import t

df = pd.read_csv("../final/problem2.csv")
x = df["SPY"]

nu, mu, sigma = t.fit(x)

print("mu\tsigma\tnu")
print(f"{mu}\t{sigma}\t{nu}")

out_df = pd.DataFrame({"nu": [nu], "mu": [mu], "sigma": [sigma]})
out_df.to_csv("../output/output7_2.csv", index=False)
