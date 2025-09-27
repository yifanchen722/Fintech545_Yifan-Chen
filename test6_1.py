import pandas as pd

data = pd.read_csv("test6.csv", parse_dates=["Date"], index_col="Date")
returns = data.pct_change().dropna()
returns.to_csv("arithmetic_returns.csv")
print(returns.head())
