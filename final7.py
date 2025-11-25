import pandas as pd
import os
import numpy as np
import statsmodels.api as sm

# ==============================
# Part A — 计算三只股票的因子 Betas
# ==============================

# Load history file
df = pd.read_csv("../data/ec1_history.csv")

Rf = 0.003  # 0.3% per month
stocks = ["S1", "S2", "S3"]

results = {}

for s in stocks:
    # 股票超额收益
    y = df[s] - Rf

    # 因子收益（已是超额收益）
    X = df[["F1", "F2"]]
    X = sm.add_constant(X)

    # 回归
    model = sm.OLS(y, X).fit()
    results[s] = model.params

# 转成 DataFrame：列为 const, F1, F2
betas = pd.DataFrame(results).T
print("=== Factor Betas ===")
print(betas)

# 保存结果到桌面
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "betas.xlsx")
betas.to_excel(desktop_path, index=True)
print("Betas saved to:", desktop_path)


# ===========================================
# Part B — 利用 forward returns 做因子归因分析
# ===========================================

# 初始组合权重
w = np.array([0.3, 0.45, 0.25])

# Load forward data
df_fwd = pd.read_csv("../data/ec1_fwd.csv")

# 股票超额收益
stock_excess = df_fwd[stocks] - Rf  # 每列减 Rf
# 因子超额收益（题目给出已是超额收益）
factor_returns = df_fwd[["F1", "F2"]]

# =============== 组合真实收益（ground truth） ===============
# 组合超额收益时间序列
portfolio_excess = stock_excess.dot(w)

# 组合真实均值 + 标准差
port_mean = portfolio_excess.mean()
port_std = portfolio_excess.std()


# =============== 利用 betas 做因子归因 ===============

# 把 betas 排序为矩阵形式
beta_matrix = betas[["F1", "F2"]].values  # shape (3 stocks × 2 factors)

# 组合的整体因子 beta
portfolio_beta = w @ beta_matrix  # (1×3) * (3×2) = (1×2)
portfolio_beta

# 组合因子贡献时间序列
factor_contrib_ts = factor_returns.mul(portfolio_beta)

# 组合因子贡献均值 + 标准差
factor_mean = factor_contrib_ts.mean()
factor_std = factor_contrib_ts.std()

# =============== 残差归因（组合 Alpha）==============

# α_t = 组合超额收益 − ∑(beta_f * factor_return_f)
alpha_ts = portfolio_excess - factor_contrib_ts.sum(axis=1)

alpha_mean = alpha_ts.mean()
alpha_std = alpha_ts.std()


# =============== 输出结果 ===============
print("\n=== Portfolio Attribution Results ===")

print("\nPortfolio Excess Return Mean:", port_mean)
print("Portfolio Excess Return Std :", port_std)

print("\nFactor Mean Contributions:")
print(factor_mean)

print("\nFactor Std Contributions:")
print(factor_std)

print("\nPortfolio Alpha Mean:", alpha_mean)
print("Portfolio Alpha Std :", alpha_std)
