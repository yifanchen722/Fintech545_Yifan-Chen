import pandas as pd
import numpy as np
from scipy.stats import norm, t, multivariate_normal

portfolio = pd.read_csv("../data/test9_1_portfolio.csv")
returns = pd.read_csv("../data/test9_1_returns.csv")

holdings = {"A": 100, "B": 100}
prices = {"A": 20, "B": 30}
initial_values = {s: holdings[s] * prices[s] for s in ["A", "B"]}

mu_A, sigma_A = returns["A"].mean(), returns["A"].std()
df_B, loc_B, scale_B = t.fit(returns["B"])

u_A = norm.cdf(returns["A"], mu_A, sigma_A)
u_B = t.cdf(returns["B"], df_B, loc_B, scale_B)
z = norm.ppf(np.column_stack([u_A, u_B]))

corr = np.corrcoef(z, rowvar=False)
n_sim = 100000
sim_z = multivariate_normal.rvs(mean=[0, 0], cov=corr, size=n_sim)
sim_u = norm.cdf(sim_z)

sim_A = norm.ppf(sim_u[:, 0], mu_A, sigma_A)
sim_B = t.ppf(sim_u[:, 1], df_B, loc_B, scale_B)

loss_A = -sim_A * initial_values["A"]
loss_B = -sim_B * initial_values["B"]
total_loss = loss_A + loss_B

alpha = 0.95
var_A, es_A = (
    np.quantile(loss_A, alpha),
    loss_A[loss_A >= np.quantile(loss_A, alpha)].mean(),
)
var_B, es_B = (
    np.quantile(loss_B, alpha),
    loss_B[loss_B >= np.quantile(loss_B, alpha)].mean(),
)
var_total, es_total = (
    np.quantile(total_loss, alpha),
    total_loss[total_loss >= np.quantile(total_loss, alpha)].mean(),
)

results = pd.DataFrame(
    {
        "Stock": ["A", "B", "Total"],
        "VaR95": [var_A, var_B, var_total],
        "ES95": [es_A, es_B, es_total],
        "VaR95_Pct": [
            var_A / initial_values["A"],
            var_B / initial_values["B"],
            var_total / (initial_values["A"] + initial_values["B"]),
        ],
        "ES95_Pct": [
            es_A / initial_values["A"],
            es_B / initial_values["B"],
            es_total / (initial_values["A"] + initial_values["B"]),
        ],
    }
)
print(results)
