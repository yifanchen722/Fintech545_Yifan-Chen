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


# Minimize SSE
def rp_objective(w, Sigma):
    CSD = csd(w, Sigma)
    target = np.mean(CSD)
    return np.sum((CSD - target) ** 2)


# Inverse Vol
init_w = 1 / np.sqrt(np.diag(Sigma))
init_w = init_w / np.sum(init_w)

# Constraints & Bounds
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * n

result = minimize(
    fun=rp_objective,
    x0=init_w,
    args=(Sigma,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)

w_rp = result.x

print("Risk Parity Weights:")
for wi in w_rp:
    print(wi)
