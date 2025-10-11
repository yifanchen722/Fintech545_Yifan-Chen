import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.stats import spearmanr

portfolio_path = "../data/test9_1_portfolio.csv"
returns_path = "../data/test9_1_returns.csv"
output_path = "/Users/nico/Desktop/output_test9_1_copula_spearman.csv"

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

# spearman_corr_df = df_returns.corr(method="spearman")
# corr_mat = spearman_corr_df.values

varcov = np.array([[sigma_X**2, cov_s], [cov_s, sigma_Y**2]])


eps = 1e-20
corr_mat = varcov + np.eye(n_assets) * eps
L = np.linalg.cholesky(corr_mat)

Z = np.random.normal(size=(K, n_assets))
Z_corr = L @ Z.T + mean_matrix
U = norm.cdf(Z_corr)
U = U.T
print("shape of U:", U.shape)
# U = np.clip(U, 1e-12, 1 - 1e-12)

sim_returns = np.empty_like(U)
print("sim_returns shape:", sim_returns.shape)
for j, a in enumerate(assets):
    stock = a["Stock"]
    params = marginal_params[stock]
    if params["dist"] == "normal":
        sim_returns[:, j] = norm.ppf(U[:, j], loc=params["mu"], scale=params["sigma"])
    elif params["dist"] == "t":
        sim_returns[:, j] = t.ppf(
            U[:, j], df=params["nu"], loc=params["loc"], scale=params["scale"]
        )

print("Simulated returns (first 5 rows):\n", sim_returns[:, :5])

positions = np.array([a["Holding"] * a["Starting Price"] for a in assets])
print("Positions:", positions)

losses_sim = np.empty_like(sim_returns)
losses_sim[:, 0] = -sim_returns[:, 0] * positions[0]
losses_sim[:, 1] = -sim_returns[:, 1] * positions[1]
# losses_sim = losses_sim.T
print("losses shape: ", losses_sim.shape)
print("losses: ", losses_sim[:5, :])

results = []
for j, a in enumerate(assets):
    pos_value = positions[j]
    loss_vec = losses_sim[:, j]

    VaR95 = np.percentile(loss_vec, 100 * (1 - alpha))
    tail = loss_vec[loss_vec > VaR95]
    ES95 = tail.mean() if len(tail) > 0 else VaR95

    results.append(
        {
            "Stock": a["Stock"],
            "VaR95": VaR95,
            "ES95": ES95,
            "VaR95_Pct": VaR95 / pos_value,
            "ES95_Pct": ES95 / pos_value,
        }
    )

total_losses = losses_sim.sum(axis=1)
VaR95_total = np.percentile(total_losses, 100 * (1 - alpha))
tail_total = total_losses[total_losses > VaR95_total]
ES95_total = tail_total.mean() if len(tail_total) > 0 else VaR95_total

total_value = positions.sum()
results.append(
    {
        "Stock": "Total",
        "VaR95": VaR95_total,
        "ES95": ES95_total,
        "VaR95_Pct": VaR95_total / total_value,
        "ES95_Pct": ES95_total / total_value,
    }
)

out_df = pd.DataFrame(results)[["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]]
out_df.to_csv(output_path, index=False)
print("\n=== VaR/ES on 2 levels (Spearman Copula) ===\n")
print(out_df)
