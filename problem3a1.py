import numpy as np
import pandas as pd
from scipy.optimize import minimize

df_in = pd.read_csv("../final/problem3_insample.csv")

df_in = df_in.drop(columns=["Date"])

assets = df_in.columns.tolist()
n = len(assets)

mean = df_in.mean().values
rf = 0.04
mu_excess = mean - rf

Sigma = df_in.cov().values


def neg_sharpe(w, mu_excess, Sigma):
    port_excess_ret = w @ mu_excess
    port_vol = np.sqrt(w @ Sigma @ w)
    return -port_excess_ret / port_vol


constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

eps = 1e-10
bounds = [(-1.0, 1.0) for _ in range(n)]

w0 = np.ones(n) / n

result = minimize(
    fun=neg_sharpe,
    x0=w0,
    args=(mu_excess, Sigma),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 10000},
)

w_opt = result.x

print("Max Sharpe Ratio Weights:")
for name, wi in zip(assets, w_opt):
    print(f"{name}: {wi:.12f}")
