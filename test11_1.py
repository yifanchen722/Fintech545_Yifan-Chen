import pandas as pd
import numpy as np
import statsmodels.api as sm

file_path = "../data/test11_1_returns.csv"
df = pd.read_csv(file_path)


assets = ["x1", "x2", "x3"]

w = np.zeros((df.shape[0], 3))

w1 = 0.3
w2 = 0.2
w3 = 0.5
R_vec = np.zeros(df.shape[0])

for t in range(df.shape[0]):
    w[t, 0] = w1
    w[t, 1] = w2
    w[t, 2] = w3

    w1 = w1 * (1 + df["x1"][t])
    w2 = w2 * (1 + df["x2"][t])
    w3 = w3 * (1 + df["x3"][t])
    Rt = (w1 + w2 + w3) - 1
    R_vec[t] = Rt

    w1 = w1 / (Rt + 1)
    w2 = w2 / (Rt + 1)
    w3 = w3 / (Rt + 1)


R = (R_vec + 1).prod() - 1

GR = np.log(1 + R)

K = GR / R

df["kt"] = np.log(1 + R_vec) / (K * R_vec)

return_attr = {}

for i, a in enumerate(assets):
    # print(i, a)
    contrib = df["kt"] * w[:, i] * df[a]
    return_attr[a] = contrib.sum()

return_attr["Portfolio"] = R


vol_attr = {}

portfolio_std = np.std(R_vec, ddof=1)

for i, a in enumerate(assets):
    y_i = w[:, i] * df[a]

    X = sm.add_constant(R_vec)
    model = sm.OLS(y_i, X).fit()

    beta = model.params["x1"]

    vol_attr[a] = portfolio_std * beta

vol_attr["Portfolio"] = portfolio_std

total_return = {a: (df[a] + 1).prod() - 1 for a in assets}
total_return["Portfolio"] = R

result = pd.DataFrame(
    {
        "x1": [total_return["x1"], return_attr["x1"], vol_attr["x1"]],
        "x2": [total_return["x2"], return_attr["x2"], vol_attr["x2"]],
        "x3": [total_return["x3"], return_attr["x3"], vol_attr["x3"]],
        "Portfolio": [
            total_return["Portfolio"],
            return_attr["Portfolio"],
            vol_attr["Portfolio"],
        ],
    },
    index=["TotalReturn", "Return Attribution", "Vol Attribution"],
)

print(result)
