import pandas as pd
import numpy as np
import statsmodels.api as sm

# === 读取 insample 数据 ===
df = pd.read_csv("../final/problem3_insample.csv")

# 去掉 Date 列
df = df.drop(columns=["Date"])

assets = df.columns.tolist()  # ["GOOG","JPM","WMT","AMD","NKE"]
N = len(assets)
T = df.shape[0]

# === 初始权重（等权或题目给定）===
weights_initial = np.ones(N) / N  # 每个资产 20%
w = np.zeros((T, N))

# === 动态更新权重 ===
w_current = weights_initial.copy()
R_vec = np.zeros(T)

for t in range(T):

    w[t] = w_current

    # 组合当期收益
    Rt = np.dot(w_current, df.iloc[t].values)
    R_vec[t] = Rt

    # 更新权重前的数值
    w_star = w_current * (1 + df.iloc[t].values)

    # 归一化
    w_current = w_star / (1 + Rt)


# === 组合总收益 ===
R_total = (R_vec + 1).prod() - 1
GR = np.log(1 + R_total)
K = GR / R_total

df["kt"] = np.log(1 + R_vec) / (K * R_vec)


# === 1. Return Attribution ===
return_attr = {}

for i, a in enumerate(assets):
    contrib = df["kt"] * w[:, i] * df[a]
    return_attr[a] = contrib.sum()

return_attr["Portfolio"] = R_total


# === 2. Volatility Attribution ===
vol_attr = {}
portfolio_std = np.std(R_vec, ddof=1)

for i, a in enumerate(assets):
    y_i = w[:, i] * df[a]

    X = sm.add_constant(R_vec)
    model = sm.OLS(y_i, X).fit()

    beta_ip = model.params[1]  # 正确取回归系数（第二个）
    vol_attr[a] = portfolio_std * beta_ip

vol_attr["Portfolio"] = portfolio_std


# === Total Return (individual assets) ===
total_return = {a: (df[a] + 1).prod() - 1 for a in assets}
total_return["Portfolio"] = R_total


# === Output ===
result = pd.DataFrame(
    {a: [total_return[a], return_attr[a], vol_attr[a]] for a in assets},
    index=["TotalReturn", "Return Attribution", "Vol Attribution"],
)

print(result)
