import pandas as pd

file_path = "test7_1.csv"
col_name = "x1"

df = pd.read_csv(file_path)
mu = df[col_name].mean()
sigma = df[col_name].std(ddof=1)

mu_str = f"{mu:.18f}"
sigma_str = f"{sigma:.17f}"

print("mu:", mu_str)
print("sigma:", sigma_str)

out_df = pd.DataFrame({"mu": [mu_str], "sigma": [sigma_str]})
out_df.to_csv("testout.csv", index=False)
