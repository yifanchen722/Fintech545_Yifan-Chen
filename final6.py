import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from scipy import optimize

# ======================================================
# 1. Load data
# ======================================================
df = pd.read_csv("../data/question6.csv")

# compute log returns
logret = np.log(df[["SPY", "AMZN", "AAPL"]] / df[["SPY", "AMZN", "AAPL"]].shift(1))
logret = logret.dropna()

# ======================================================
# 2. Structural regressions
# ======================================================
X = sm.add_constant(logret["SPY"])

res_aapl = sm.OLS(logret["AAPL"], X).fit()
res_amzn = sm.OLS(logret["AMZN"], X).fit()

alpha1, beta1 = res_aapl.params
alpha2, beta2 = res_amzn.params

eps1 = res_aapl.resid.values
eps2 = res_amzn.resid.values


# ======================================================
# 3. Fit Student-t distribution to residuals
# ======================================================
def fit_student_t(data):

    # negative log-likelihood
    def nll(params):
        df, loc, scale = params
        if scale <= 0 or df <= 2:
            return 1e10
        return -np.sum(t.logpdf(data, df=df, loc=loc, scale=scale))

    x0 = np.array([10, 0, np.std(data)])  # initial guess
    bounds = [(2.1, 200), (-1, 1), (1e-6, None)]  # df > 2 so variance finite
    result = optimize.minimize(nll, x0, bounds=bounds)

    df_est, loc_est, scale_est = result.x
    return df_est, loc_est, scale_est


df1, loc1, scale1 = fit_student_t(eps1)
df2, loc2, scale2 = fit_student_t(eps2)

# assume iid → same df, use average df
df_t = (df1 + df2) / 2

# portfolio's Student-t scale for idiosyncratic shocks
# ======================================================
last_aapl = df["AAPL"].iloc[-1]
last_amzn = df["AMZN"].iloc[-1]

PV = 100 * last_aapl + 100 * last_amzn

w_aapl = (100 * last_aapl) / PV
w_amzn = (100 * last_amzn) / PV

alpha_p = w_aapl * alpha1 + w_amzn * alpha2
beta_p = w_aapl * beta1 + w_amzn * beta2

# portfolio idiosyncratic scale
scale_p = np.sqrt((w_aapl**2) * (scale1**2) + (w_amzn**2) * (scale2**2))

# ======================================================
# 4. SPY mean & variance
# ======================================================
mu_spy = logret["SPY"].mean()
var_spy = logret["SPY"].var()

mu_p = alpha_p + beta_p * mu_spy
sigma_p = np.sqrt(beta_p**2 * var_spy + scale_p**2)

# ======================================================
# 5. Student-t VaR (95%)
# ======================================================
t_005 = t.ppf(0.05, df_t)

VaR_95_t = -(mu_p + t_005 * sigma_p) * PV

print("Student-t Degrees of Freedom =", df_t)
print("Portfolio 95% VaR (Student-t) = $", VaR_95_t)
