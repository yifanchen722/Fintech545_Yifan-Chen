import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import brentq

# ============================================================
# 1. Read data & compute arithmetic returns
# ============================================================

df = pd.read_csv("../GitHub/problem6.csv")
prices_A = df["AssetA"].values
prices_B = df["AssetB"].values

rA = (prices_A[1:] - prices_A[:-1]) / prices_A[:-1]
rB = (prices_B[1:] - prices_B[:-1]) / prices_B[:-1]

returns = np.vstack([rA, rB]).T
Sigma = np.cov(returns.T)

S_A0 = prices_A[-1]
S_B0 = prices_B[-1]

rf = 0.0475
rf_daily = rf / 252

# ============================================================
# 2. Black–Scholes option pricing helpers
# ============================================================


def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, r, sigma, T, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


def bs_delta_put(S, K, r, sigma, T, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def bs_delta_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ============================================================
# 3. Implied volatility solver for Asset B call
# ============================================================

K_B = 100
T_B = 100 / 252
C0_B = 6.50


def call_error(sigma):
    return bs_call(S_B0, K_B, rf, sigma, T_B) - C0_B


sigma_B = brentq(lambda x: call_error(x), 1e-6, 3.0)

print("Implied vol for Asset B =", sigma_B)

# ============================================================
# 4. Portfolio Delta exposures for Delta-Normal VaR
# ============================================================

# Asset A (Put + stock)
K_A = 100
T_A = 1
sigma_A = 0.20
q_A = 0  # treating dividends approximately as continuous 0

delta_put_A = bs_delta_put(S_A0, K_A, rf, sigma_A, T_A, q_A)
Delta_A = 100 + 100 * delta_put_A

# Asset B (Call short + stock)
delta_call_B = bs_delta_call(S_B0, K_B, rf, sigma_B, T_B)
Delta_B = 50 - 50 * delta_call_B

# Dollar exposures
w = np.array([Delta_A * S_A0, Delta_B * S_B0])

# ============================================================
# 5. Delta-Normal VaR & ES
# ============================================================

portfolio_var = w.T @ Sigma @ w
portfolio_std = np.sqrt(portfolio_var)

z95 = norm.ppf(0.95)
VaR_delta = z95 * portfolio_std
ES_delta = portfolio_std * norm.pdf(z95) / 0.05

print("\n=== Delta-Normal ===")
print("VaR (5%):", VaR_delta)
print("ES  (5%):", ES_delta)

# ============================================================
# 6. Monte Carlo (Normal) VaR & ES
# ============================================================

N = 50000
R = np.random.multivariate_normal([0, 0], Sigma, size=N)


def price_portfolio(SA, SB):
    putA = bs_put(SA, K_A, rf, sigma_A, T_A)
    callB = bs_call(SB, K_B, rf, sigma_B, T_B)
    return 100 * SA + 100 * putA + 50 * SB - 50 * callB


V0 = price_portfolio(S_A0, S_B0)

losses = []
for i in range(N):
    SA = S_A0 * (1 + R[i, 0])
    SB = S_B0 * (1 + R[i, 1])
    Vt = price_portfolio(SA, SB)
    losses.append(V0 - Vt)

losses = np.sort(losses)
VaR_mc = losses[int(0.95 * N)]
ES_mc = losses[int(0.95 * N) :].mean()

print("\n=== Monte Carlo (Normal) ===")
print("VaR (5%):", VaR_mc)
print("ES  (5%):", ES_mc)

# ============================================================
# 7. Best-Fit Model (Normal vs t) + Copula Monte Carlo
# ============================================================

# Fit normal
muA, stdA = np.mean(rA), np.std(rA)
muB, stdB = np.mean(rB), np.std(rB)


# Fit t distribution
def fit_t(x):
    # simple MLE via moments
    nu = 5  # you may replace with numeric MLE if needed
    s = np.std(x) * np.sqrt((nu - 2) / nu)
    return nu, s, np.mean(x)


nuA, sA, muA_t = fit_t(rA)
nuB, sB, muB_t = fit_t(rB)


# Choose better based on kurtosis
def choose_model(x):
    if np.abs(pd.Series(x).kurt()) > 3:
        return "t"
    return "normal"


modelA = choose_model(rA)
modelB = choose_model(rB)

print("\nAsset A uses:", modelA)
print("Asset B uses:", modelB)

# Gaussian copula + inverse CDF
corr = np.corrcoef(rA, rB)[0, 1]
R_cop = np.array([[1, corr], [corr, 1]])

Z = np.random.multivariate_normal([0, 0], R_cop, size=N)

U = norm.cdf(Z)


def inv_marginal(u, model, mu, sigma, nu=None):
    if model == "normal":
        return mu + sigma * norm.ppf(u)
    else:
        return mu + sigma * t.ppf(u, df=nu)


losses2 = []
for i in range(N):
    uA, uB = U[i]
    # asset A
    if modelA == "normal":
        rAi = inv_marginal(uA, "normal", muA, stdA)
    else:
        rAi = inv_marginal(uA, "t", muA_t, sA, nuA)
    # asset B
    if modelB == "normal":
        rBi = inv_marginal(uB, "normal", muB, stdB)
    else:
        rBi = inv_marginal(uB, "t", muB_t, sB, nuB)

    SA = S_A0 * (1 + rAi)
    SB = S_B0 * (1 + rBi)
    Vt = price_portfolio(SA, SB)
    losses2.append(V0 - Vt)

losses2 = np.sort(losses2)
VaR_best = losses2[int(0.95 * N)]
ES_best = losses2[int(0.95 * N) :].mean()

print("\n=== Best-Fit (Normal/t) + Copula MC ===")
print("VaR (5%):", VaR_best)
print("ES  (5%):", ES_best)
