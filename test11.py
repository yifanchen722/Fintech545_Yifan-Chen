import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.stats import spearmanr

portfolio_path = "../data/test9_1_portfolio.csv"
returns_path = "../output/problem6_returns.csv"
output_path = "/Users/nico/Desktop/output_problem6_spearman.csv"

K = 100000
alpha = 0.05

portfolio_df = pd.read_csv(portfolio_path)
assets = portfolio_df.to_dict(orient="records")
asset_names = [a["Stock"] for a in assets]

df_returns = pd.read_csv(returns_path)
returns = df_returns[asset_names].values
n_obs, n_assets = returns.shape

marginal_params = {}
for i, a in enumerate(assets):
    stock = a["Stock"]
    r = returns[:, i]
    if a["Distribution"].lower() == "normal":
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        marginal_params[stock] = {"dist": "normal", "mu": mu, "sigma": sigma}
    elif a["Distribution"].lower() == "t":
        nu, loc, scale = t.fit(r)
        marginal_params[stock] = {"dist": "t", "nu": nu, "loc": loc, "scale": scale}
    else:
        raise ValueError(
            f"Unsupported distribution type for {stock}: {a['Distribution']}"
        )

# construct Spearman correlation matrix
a_conv = norm.ppf(
    norm.cdf(
        df_returns.A,
        loc=marginal_params["A"]["mu"],
        scale=marginal_params["A"]["sigma"],
    )
)
b_conv = norm.ppf(
    t.cdf(
        df_returns.B,
        df=marginal_params["B"]["nu"],
        loc=marginal_params["B"]["loc"],
        scale=marginal_params["B"]["scale"],
    )
)

mean_a_conv = np.mean(a_conv)
mean_b_conv = np.mean(b_conv)
print("Means of transformed data:", mean_a_conv, mean_b_conv)
mean_matrix = np.tile(np.array([[mean_a_conv], [mean_b_conv]]), (1, K))

rho, _ = spearmanr(a_conv, b_conv)
print("Spearman correlation:", rho)

sigma_X = np.std(a_conv, ddof=1)
sigma_Y = np.std(b_conv, ddof=1)

cov_s = rho * sigma_X * sigma_Y
print("Spearman covariance:", cov_s)
