import numpy as np
import pandas as pd
import statsmodels.api as sm

file_path_stockreturn = "../data/test11_2_stock_returns.csv"
df_stock = pd.read_csv(file_path_stockreturn)

r1 = df_stock["S1"].values
r2 = df_stock["S2"].values

file_path_factorreturn = "../data/test11_2_factor_returns.csv"
df_factor = pd.read_csv(file_path_factorreturn)

F1 = df_factor["F1"].values
F2 = df_factor["F2"].values
F3 = df_factor["F3"].values

beta1 = np.array([0.6386451485061868, 0.3739575749535735, -0.22506858048271394])
beta2 = np.array([0.9813995072495585, -0.3825387461932517, 0.9897016030357082])

w1_initial = 0.5
w2_initial = 0.5

T = len(r1)

w1 = np.zeros(T)
w2 = np.zeros(T)
Rp = np.zeros(T)

w1[0] = w1_initial
w2[0] = w2_initial

for t in range(T):
    Rp[t] = w1[t] * r1[t] + w2[t] * r2[t]
    if t < T - 1:
        w1_star = w1[t] * (1 + r1[t])
        w2_star = w2[t] * (1 + r2[t])

        w1[t + 1] = w1_star / (1 + Rp[t])
        w2[t + 1] = w2_star / (1 + Rp[t])

wF = np.zeros((T, 3))

for t in range(T):
    wF[t, 0] = w1[t] * beta1[0] + w2[t] * beta2[0]
    wF[t, 1] = w1[t] * beta1[1] + w2[t] * beta2[1]
    wF[t, 2] = w1[t] * beta1[2] + w2[t] * beta2[2]

F = np.vstack([F1, F2, F3]).T

factor_contrib = np.sum(wF * F, axis=1)
alpha_t = Rp - factor_contrib

factor_contrib_individual = wF * F

R_arith = np.prod(1 + Rp) - 1
GR = np.log(1 + R_arith)
K = GR / R_arith

k_t = np.log(1 + Rp) / (K * Rp)

RA_return_factor = np.sum(k_t[:, None] * factor_contrib_individual, axis=0)

RA_return_alpha = np.sum(k_t * alpha_t)

sigma_p = np.std(Rp, ddof=1)

RA_vol_factor = np.zeros(3)

for j in range(3):
    X = sm.add_constant(Rp)
    y = factor_contrib_individual[:, j]
    model = sm.OLS(y, X).fit()
    beta_jp = model.params[1]
    RA_vol_factor[j] = sigma_p * beta_jp

X = sm.add_constant(Rp)
model_alpha = sm.OLS(alpha_t, X).fit()
beta_ap = model_alpha.params[1]
RA_vol_alpha = sigma_p * beta_ap


# TotalReturn for each component
total_return_factors = [
    (F1 + 1).prod() - 1,
    (F2 + 1).prod() - 1,
    (F3 + 1).prod() - 1,
]

# Alpha total return
total_return_alpha = (alpha_t + 1).prod() - 1

# Portfolio total return
total_return_portfolio = (Rp + 1).prod() - 1


portfolio_vol_attr = RA_vol_factor.sum() + RA_vol_alpha

result = pd.DataFrame(
    {
        "F1": [total_return_factors[0], RA_return_factor[0], RA_vol_factor[0]],
        "F2": [total_return_factors[1], RA_return_factor[1], RA_vol_factor[1]],
        "F3": [total_return_factors[2], RA_return_factor[2], RA_vol_factor[2]],
        "Alpha": [total_return_alpha, RA_return_alpha, RA_vol_alpha],
        "Portfolio": [
            total_return_portfolio,
            total_return_portfolio,
            portfolio_vol_attr,
        ],
    },
    index=["TotalReturn", "Return Attribution", "Vol Attribution"],
)

print(result)
