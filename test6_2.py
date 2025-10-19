import pandas as pd
import numpy as np

data = pd.read_csv("../data/test6.csv", parse_dates=["Date"], index_col="Date")
log_returns = np.log(data / data.shift(1)).dropna()

print(log_returns.head())
