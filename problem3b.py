import pandas as pd
import numpy as np
import statsmodels.api as sm

file_path = "../final/problem3_outsample.csv"
df = pd.read_csv(file_path)
df = df.iloc[:, 1:]

assets = ["GOOG", "JPM", "WMT", "AMD", "NKE"]
n_assets = len(assets)
T = df.shape[0]

w0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
w = np.zeros((T, n_assets))
R_vec = np.zeros(T)

current_w = w0.copy()

for t in range(T):

    w[t, :] = current_w

    ret_t = df[assets].iloc[t].values

    current_value = current_w * (1 + ret_t)
    Rt = current_value.sum() - 1
    R_vec[t] = Rt

    current_w = current_value / (1 + Rt)


R = (R_vec + 1).prod() - 1
GR = np.log(1 + R)
K = GR / R


df["kt"] = np.log(1 + R_vec) / (K * R_vec)


return_attr = {}

for i, a in enumerate(assets):
    contrib = df["kt"] * w[:, i] * df[a]
    return_attr[a] = contrib.sum()

return_attr["Portfolio"] = R


vol_attr = {}
portfolio_std = np.std(R_vec, ddof=1)

for i, a in enumerate(assets):

    y_i = w[:, i] * df[a]

    X = sm.add_constant(R_vec)
    model = sm.OLS(y_i, X).fit()

    beta = model.params[1]

    vol_attr[a] = portfolio_std * beta

vol_attr["Portfolio"] = portfolio_std


total_return = {a: (df[a] + 1).prod() - 1 for a in assets}
total_return["Portfolio"] = R


result = pd.DataFrame(
    {a: [total_return[a], return_attr[a], vol_attr[a]] for a in assets + ["Portfolio"]},
    index=["TotalReturn", "Return Attribution", "Vol Attribution"],
)

print(result)
