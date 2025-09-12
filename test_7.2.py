import pandas as pd
from scipy.stats import t

df = pd.read_csv('test7_2.csv')
x = df['x1']

nu, mu, sigma = t.fit(x)

print("mu\tsigma\tnu")
print(f"{mu:.17f}\t{sigma:.17f}\t{nu:.15f}")
