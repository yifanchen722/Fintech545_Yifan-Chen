import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st

df = pd.read_csv("../data/question6.csv")


# ================================
# 2. 计算 log return（对数收益）
# ================================
# 对价格取 log，再做差分
logret = np.log(df[["SPY", "AMZN", "AAPL"]]) - np.log(
    df[["SPY", "AMZN", "AAPL"]].shift(1)
)
logret = logret.dropna()

# ================================
# 3. 回归结构模型：AAPL ~ SPY
# ================================
X = sm.add_constant(logret["SPY"])  # 添加常数项
model_aapl = sm.OLS(logret["AAPL"], X).fit()

alpha1, beta1 = model_aapl.params
var_eps1 = np.var(model_aapl.resid, ddof=1)

print("AAPL 回归结果：")
print(model_aapl.summary())

# ================================
# 4. 回归结构模型：AMZN ~ SPY
# ================================
model_amzn = sm.OLS(logret["AMZN"], X).fit()

alpha2, beta2 = model_amzn.params
var_eps2 = np.var(model_amzn.resid, ddof=1)

print("\nAMZN 回归结果：")
print(model_amzn.summary())

# ================================
# 5. 计算投资组合价值与权重
# ================================
last_aapl = df["AAPL"].iloc[-1]
last_amzn = df["AMZN"].iloc[-1]

# 100 股 AAPL + 100 股 AMZN
PV = 100 * last_aapl + 100 * last_amzn

w_aapl = (100 * last_aapl) / PV
w_amzn = (100 * last_amzn) / PV

print("\n组合价值 PV =", PV)
print("权重 AAPL =", w_aapl)
print("权重 AMZN =", w_amzn)

# ================================
# 6. 组合的 alpha、beta
# ================================
alpha_p = w_aapl * alpha1 + w_amzn * alpha2
beta_p = w_aapl * beta1 + w_amzn * beta2

# ================================
# 7. 组合误差项的方差
# ================================
var_eps_p = w_aapl**2 * var_eps1 + w_amzn**2 * var_eps2

# ================================
# 8. SPY 的均值与方差
# ================================
mu_spy = logret["SPY"].mean()
var_spy = logret["SPY"].var()

# ================================
# 9. 组合收益分布参数
# ================================
mu_p = alpha_p + beta_p * mu_spy
var_p = beta_p**2 * var_spy + var_eps_p
sigma_p = np.sqrt(var_p)

print("\n组合收益均值 mu_p =", mu_p)
print("组合收益方差 var_p =", var_p)
print("组合收益波动 sigma_p =", sigma_p)

# ================================
# 10. 计算 95% VaR（正态分布）
# ================================
z = st.norm.ppf(0.05)  # 5% 分位数
VaR_95 = -(mu_p + z * sigma_p) * PV

print("\n==============================")
print("组合的一日 95% VaR =", VaR_95)
print("==============================")
