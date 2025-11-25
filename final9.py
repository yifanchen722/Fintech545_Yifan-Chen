import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ================================
# Black-Scholes Put
# ================================
def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put


# ================================
# (a) Implied Volatility Solve
# ================================
S0 = 100
K = 100
T = 1.0
r = 0.02
market_price = 7.18


def f(sig):
    return bs_put(S0, K, T, r, sig) - market_price


# 用 Brent 求根 (最稳定)
implied_vol = brentq(f, 1e-10, 3.0)
print("Implied Volatility =", implied_vol)

# ================================
# (b) 1-day VaR & ES
# ================================
mu = 0.04  # 年化期望收益
sigma = implied_vol
days = 255
dt = 1 / days
N = 300000  # Monte Carlo 路径

# 当前期权价格（应≈7.18）
P0 = bs_put(S0, K, T, r, sigma)

# ---- 向量化模拟 S1 ----
Z = np.random.randn(N)
S1 = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
T1 = T - dt

# ---- 向量化重新定价期权（关键：无循环） ----
d1 = (np.log(S1 / K) + (r + 0.5 * sigma * sigma) * T1) / (sigma * np.sqrt(T1))
d2 = d1 - sigma * np.sqrt(T1)
P1 = K * np.exp(-r * T1) * norm.cdf(-d2) - S1 * norm.cdf(-d1)

# 空头损失
loss = P1 - P0


# ---- VaR / ES ----
def var_es(loss_array, alpha=0.99):
    VaR = np.percentile(loss_array, alpha * 100)
    ES = loss_array[loss_array > VaR].mean()
    return VaR, ES


VaR_95, ES_95 = var_es(loss, 0.95)
VaR_99, ES_99 = var_es(loss, 0.99)

print("---- 1-Day Risk Metrics ----")
print("VaR 95% =", VaR_95)
print("ES 95%  =", ES_95)
print("VaR 99% =", VaR_99)
print("ES 99%  =", ES_99)
