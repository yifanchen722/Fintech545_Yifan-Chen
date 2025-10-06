import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


data = pd.read_csv("../data/test7_3.csv")
X = data[["x1", "x2", "x3"]].values
y = data["y"].values
n = len(y)
X = np.column_stack([np.ones(n), X])


def t_loglikelihood(params, X, y):
    alpha = params[0]
    B = params[1:4]
    sigma = np.exp(params[4])
    nu = np.exp(params[5])
    mu = alpha + X[:, 1:] @ B
    xm = ((y - mu) / sigma) ** 2
    np12 = (nu + 1) / 2
    ll = np.sum(
        gammaln(np12)
        - gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
        - np12 * np.log(1 + xm / nu)
    )
    return -ll


init_params = np.array([0.0, 0.0, 0.0, 0.0, np.log(0.05), np.log(5.0)])
result = minimize(t_loglikelihood, init_params, args=(X, y), method="L-BFGS-B")

alpha_hat = result.x[0]
B_hat = result.x[1:4]
sigma_hat = np.exp(result.x[4])
nu_hat = np.exp(result.x[5])
mu_hat = 0.0

print("mu\tsigma\tnu\tAlpha\tB1\tB2\tB3")
print(
    f"{mu_hat}\t{sigma_hat}\t{nu_hat}\t{alpha_hat}\t{B_hat[0]}\t{B_hat[1]}\t{B_hat[2]}"
)

out_df = pd.DataFrame(
    {
        "mu_hat": [mu_hat],
        "sigma_hat": [sigma_hat],
        "nu_hat": [nu_hat],
        "alpha_hat": [alpha_hat],
        "B_hat[0]": [B_hat[0]],
        "B_hat[1]": [B_hat[1]],
        "B_hat[2]": [B_hat[2]],
    }
)
out_df.to_csv("../output/output7_3.csv", index=False)
