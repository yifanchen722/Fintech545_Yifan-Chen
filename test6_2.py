import pandas as pd
import numpy as np

data = pd.read_csv("test6.csv", parse_dates=["Date"], index_col="Date")
log_returns = np.log(data / data.shift(1)).dropna()
log_returns.to_csv("log_returns.csv")
print(log_returns.head())
