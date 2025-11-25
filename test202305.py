import numpy as np
import pandas as pd
from scipy import stats

# ======================
# 1. 读取数据
# ======================
df = pd.read_csv("problem5.csv")
prices = df[["A", "B", "C", "D"]]

# 计算算术收益
rets = prices.pct_change().dropna()

# ======================
# 2. 拟合 Student-t 分布（可视为 generalized t）
# ======================
fit_params = {}

for col in rets.columns:
    data = rets[col].values
    df_t, loc_t, scale_t = stats.t.fit(data)
    fit_params[col] = {"df": df_t, "loc": loc_t, "scale": scale_t}

print("\n=== 拟合 Student-t 参数（df, loc, scale） ===")
for k, v in fit_params.items():
    print(k, v)


# ======================
# 3. 单资产 ES(5%) 计算
# ======================


def es_from_t(df_, loc_, scale_, alpha=0.05, n_sim=500000):
    """用模拟方法计算 ES(5%)"""
    sims = stats.t.rvs(df_, loc=loc_, scale=scale_, size=n_sim)
    losses = -sims
    var_level = np.quantile(losses, 1 - alpha)
    es = losses[losses >= var_level].mean()
    return es, var_level


asset_es = {}
for col, pars in fit_params.items():
    es, var_ = es_from_t(pars["df"], pars["loc"], pars["scale"])
    asset_es[col] = {"ES_5pct_loss": es, "VaR_5pct_loss": var_}

print("\n=== 单资产 ES(5%) ===")
for k, v in asset_es.items():
    print(f"{k}: ES={v['ES_5pct_loss']:.6f}, VaR={v['VaR_5pct_loss']:.6f}")


# ======================
# 4. 构建 Gaussian Copula
# ======================

# Step 1: U = F_t(r)
U = np.zeros_like(rets.values)
cols = ["A", "B", "C", "D"]

for i, col in enumerate(cols):
    pars = fit_params[col]
    U[:, i] = stats.t.cdf(
        rets[col].values, pars["df"], loc=pars["loc"], scale=pars["scale"]
    )

# 避免极端数值
eps = 1e-6
U = np.clip(U, eps, 1 - eps)

# Step 2: 转换成 Z = Φ^{-1}(U)
Z = stats.norm.ppf(U)

# Step 3: 估计相关矩阵 Σ
Sigma = np.corrcoef(Z, rowvar=False)
print("\n=== Copula 相关矩阵 Σ ===")
print(Sigma)


# ======================
# 5. 从 Gaussian Copula + t 分布模拟联合收益
# ======================

n_sim = 500000
dim = 4
rng = np.random.default_rng(123)

# Cholesky 分解生成相关正态
L = np.linalg.cholesky(Sigma)
Z_indep = rng.standard_normal(size=(n_sim, dim))
Z_sim = Z_indep @ L.T

# 转换回 U
U_sim = stats.norm.cdf(Z_sim)

# 用单变量 t.ppf 得到最终模拟收益
R_sim = np.zeros_like(U_sim)

for i, col in enumerate(cols):
    pars = fit_params[col]
    R_sim[:, i] = stats.t.ppf(
        U_sim[:, i], pars["df"], loc=pars["loc"], scale=pars["scale"]
    )


# ======================
# 6. 组合 ES(5%) — 使用 1 股权重（按初始价格比例）
# ======================

p0 = prices.iloc[0]  # 取第一个价格作为“初始价值”


def portfolio_es_from_sims(R_sims, asset_indices, alpha=0.05):
    """给定资产编号列表（如 [0,1]），计算组合 ES(5%)"""
    p_init = p0.iloc[asset_indices].values
    w = p_init / p_init.sum()
    port_ret = R_sims[:, asset_indices] @ w
    losses = -port_ret
    var_level = np.quantile(losses, 1 - alpha)
    es = losses[losses >= var_level].mean()
    return es, var_level, w


# A & B
es_ab, var_ab, w_ab = portfolio_es_from_sims(R_sim, [0, 1])

# C & D
es_cd, var_cd, w_cd = portfolio_es_from_sims(R_sim, [2, 3])

# ALL 4
es_all, var_all, w_all = portfolio_es_from_sims(R_sim, [0, 1, 2, 3])


# ======================
# 7. 打印最终结果
# ======================

print("\n=== 组合 ES(5%) 结果 ===")

print(f"Portfolio A+B: ES={es_ab:.6f}, VaR={var_ab:.6f}, weights={w_ab}")
print(f"Portfolio C+D: ES={es_cd:.6f}, VaR={var_cd:.6f}, weights={w_cd}")
print(f"All 4 assets: ES={es_all:.6f}, VaR={var_all:.6f}, weights={w_all}")

print("\n=== 程序运行完毕 ===")
