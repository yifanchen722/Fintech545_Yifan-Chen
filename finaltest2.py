import numpy as np
import pandas as pd
from scipy.optimize import minimize

# === 读取 insample 数据 ===
df_in = pd.read_csv("../final/problem3_insample.csv")

# 去掉日期列
df_in = df_in.drop(columns=["Date"])

assets = df_in.columns.tolist()
n = len(assets)

# === Step 1: 计算 μ（均值）和 Σ（协方差） ===
mean = df_in.mean().values  # shape (n,)
rf = 0.04  # 题目给的无风险利率
mu_excess = mean - rf  # 超额收益

Sigma = df_in.cov().values  # 协方差矩阵 (n×n)


# === Step 2: 定义最大夏普率目标函数 ===
def neg_sharpe(w, mu_excess, Sigma):
    port_excess_ret = w @ mu_excess
    port_vol = np.sqrt(w @ Sigma @ w)
    return -port_excess_ret / port_vol


# === Step 3: 约束：权重之和为1 ===
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

# === Step 4: 权重限制：long only（>=0）===
eps = 1e-10
bounds = [(eps, 1.0) for _ in range(n)]

# 初始权重
w0 = np.ones(n) / n

# === Step 5: 优化求解 ===
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

# === 打印结果 ===
print("Max Sharpe Ratio Weights:")
for name, wi in zip(assets, w_opt):
    print(f"{name}: {wi:.12f}")


# outsample 回测
df_out = pd.read_csv("../final/problem3_outsample.csv")
df_out = df_out.drop(columns=["Date"])

# 组合 outsample 收益
R_out = df_out.values @ w_opt

# 累积收益
total_outsample_return = (1 + R_out).prod() - 1
print("Out-of-sample return:", total_outsample_return)
