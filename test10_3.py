import numpy as np
import pandas as pd
from scipy.optimize import minimize

file_path = "../data/test5_2.csv"
Sigma = pd.read_csv(file_path, header=0).values
n = Sigma.shape[0]

mean = np.array([0.09, 0.08, 0.07, 0.06, 0.05])
r_f = 0.04

mu_excess = mean - r_f


def neg_sharpe(w, mu_excess, Sigma):
    port_excess_ret = w @ mu_excess
    port_vol = np.sqrt(w @ Sigma @ w)
    return -port_excess_ret / port_vol


constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

eps = 1e-10
bounds = [(eps, 1.0) for _ in range(n)]

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
for wi in w_opt:
    print(f"{wi:.12f}")
