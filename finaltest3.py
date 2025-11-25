import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv("../final/problem3_insample.csv")
# 假设列名就是 5 只股票的名字
# df = df.iloc[:, 1:]
# prices = df
rets = df.iloc[:, 1:]

# 计算收益率，针对数据是price的情况
# rets = prices.pct_change().dropna()


# 期望收益 μ（5 维）
mu = rets.mean().values

# 协方差矩阵 Σ（5x5）
Sigma = rets.cov().values

r_f = 0.04
mu_excess = mu - r_f

n = len(mu)  # n=5


def neg_sharpe(w, mu_excess, Sigma):
    port_excess = w @ mu_excess
    port_vol = np.sqrt(w @ Sigma @ w)
    return -port_excess / port_vol


constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1.0) for _ in range(n)]  # 允许 w >= -1

w0 = np.ones(n) / n

res = minimize(
    neg_sharpe,
    w0,
    args=(mu_excess, Sigma),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

w_sharpe = res.x
print("\n=== 最大 Sharpe 比例组合（5资产） ===")
print(w_sharpe)


def expected_shortfall(r_p, alpha=0.05):
    r_sorted = np.sort(r_p)
    k = max(1, int(alpha * len(r_sorted)))
    tail = r_sorted[:k]
    return -tail.mean()


def neg_mu_over_es(w, rets, r_f, alpha=0.05):
    r_p = rets.values @ w
    mu_p = r_p.mean()
    es_p = expected_shortfall(r_p, alpha)

    if es_p == 0:
        return 1e9

    return -(mu_p - r_f) / es_p


constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds = [(0, 1.0) for _ in range(n)]

w0 = np.ones(n) / n

res_es = minimize(
    neg_mu_over_es,
    w0,
    args=(rets, r_f, 0.05),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

w_es = res_es.x
print("\n=== 最大 (μ-r_f)/ES 组合（5资产） ===")
print(w_es)
