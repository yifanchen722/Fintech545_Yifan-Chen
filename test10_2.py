import numpy as np
import pandas as pd
from scipy.optimize import minimize

file_path = "../data/test5_2.csv"
Sigma = pd.read_csv(file_path, header=0).values
n = Sigma.shape[0]


# CSD
def csd(w, Sigma):
    port_vol = np.sqrt(w @ Sigma @ w)
    return (w * (Sigma @ w)) / port_vol


# Objective with Non-Equal Risk Budget ----
def rp_objective_non_equal(w, Sigma, b):
    CSD = csd(w, Sigma)
    CSD_adj = CSD / b
    target = np.mean(CSD_adj)
    return np.sum((CSD_adj - target) ** 2)


# Risk Budgets: X5 has 1/2 weight ----
b = np.array([1, 1, 1, 1, 0.5])

# Inverse Volatility
init_w = 1 / np.sqrt(np.diag(Sigma))
init_w = init_w / np.sum(init_w)

# Constraints & Bounds
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * n

result = minimize(
    fun=rp_objective_non_equal,
    x0=init_w,
    args=(Sigma, b),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 2000},
)

w_rp = result.x

print("Risk Parity Weights with Non-Equal Risk Budgets:")
for wi in w_rp:
    print(wi)
