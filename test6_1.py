import pandas as pd

data = pd.read_csv("../data/test6.csv", parse_dates=["Date"], index_col="Date")
returns = data.pct_change().dropna()

print(returns.head())
