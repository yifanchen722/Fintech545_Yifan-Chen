import pandas as pd

file_path = "/Users/nico/Desktop/test7_1.csv"
col_name = "x1"

df = pd.read_csv(file_path)
mu = df[col_name].mean()
sigma = df[col_name].std(ddof=1)

mu_str = f"{mu:.18f}"
sigma_str = f"{sigma:.17f}"

print("mu:", mu_str)
print("sigma:", sigma_str)

with open("/Users/nico/Desktop/testout.csv", "w", encoding="utf-8") as f:
    f.write(f"{mu_str}\t{sigma_str}\n")
