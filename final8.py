import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================
# 1. 给定题目数据
# ============================

rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])

sigma = np.array([0.1, 0.2, 0.3])

# 协方差矩阵
cov = rho * np.outer(sigma, sigma)

# ============================
# 2. A：等权组合风险贡献
# ============================

w_eq = np.array([1 / 3, 1 / 3, 1 / 3])

portfolio_vol = np.sqrt(w_eq @ cov @ w_eq)
mrc = cov @ w_eq / portfolio_vol  # marginal risk contribution
rc = w_eq * mrc  # total risk contribution

print("A. Risk Contribution under equal weights")
print("Portfolio Vol = ", portfolio_vol)
print("Marginal RC = ", mrc)
print("Risk Contribution = ", rc)
print("RC % = ", rc / portfolio_vol)

# ============================
# 3. B：求风险平价组合
# ============================


def risk_contribution(weights, cov):
    sigma_p = np.sqrt(weights @ cov @ weights)
    mrc = cov @ weights / sigma_p
    rc = weights * mrc
    return rc


def objective(weights, cov):
    rc = risk_contribution(weights, cov)
    # 目标：让所有 RC 相等 → 最小化 RC 之间的方差
    return np.var(rc)


cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * 3
w0 = np.array([1 / 3, 1 / 3, 1 / 3])

res = minimize(
    objective, w0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons
)
w_rp = res.x

print("\nB. Risk Parity Portfolio Weights")
print("RP Weights = ", w_rp)
print("RP Risk Contributions = ", risk_contribution(w_rp, cov))
