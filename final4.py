import numpy as np
from scipy.optimize import minimize

# ----- Step 1: Define covariance matrix from problem -----

Corr = np.array([[1.0, 0.7, 0.4], [0.7, 1.0, 0.6], [0.4, 0.6, 1.0]])

vol = np.array([0.1, 0.2, 0.3])
Sigma = np.outer(vol, vol) * Corr

n = Sigma.shape[0]


# ----- Step 2: Risk parity functions -----


# Component Standard Deviation
def csd(w, Sigma):
    port_vol = np.sqrt(w @ Sigma @ w)
    return (w * (Sigma @ w)) / port_vol


# SSE objective for risk parity
def rp_objective(w, Sigma):
    CSD = csd(w, Sigma)
    target = np.mean(CSD)  # equal risk contribution
    return np.sum((CSD - target) ** 2)


# ----- Step 3: Initialize with inverse-vol weights -----

init_w = 1 / np.sqrt(np.diag(Sigma))
init_w = init_w / np.sum(init_w)


# ----- Step 4: Constraints and bounds -----

constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * n  # long-only risk parity


# ----- Step 5: Optimization -----

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
    print(round(wi, 6))
