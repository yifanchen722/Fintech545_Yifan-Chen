import numpy as np
from scipy.optimize import minimize

# Risk free rate
r_f = 0.04

# Correlation matrix
Corr = np.array([[1.0, 0.7, 0.4], [1.0, 1.0, 0.6], [0.4, 0.6, 1.0]])

# Volatilities
vol = np.array([0.1, 0.2, 0.3])

# Expected returns
mean = np.array([0.05, 0.07, 0.09])

# Covariance matrix
Sigma = np.outer(vol, vol) * Corr

# Excess returns
mu_excess = mean - r_f


# Objective: negative Sharpe ratio
def neg_sharpe(w, mu_excess, Sigma):
    port_excess = w @ mu_excess
    vol = np.sqrt(w @ Sigma @ w)
    return -port_excess / vol


# Allow short-selling: no bounds
bounds = [(None, None)] * 3

# No constraints
constraints = ()

# Initial guess
w0 = np.array([1 / 3, 1 / 3, 1 / 3])

result = minimize(
    fun=neg_sharpe,
    x0=w0,
    args=(mu_excess, Sigma),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

w_opt = result.x

print("Optimal Weights (short allowed):")
print(w_opt)
print("Max Sharpe Ratio:", -result.fun)
