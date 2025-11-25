import numpy as np
import pandas as pd
from scipy.stats import t, norm
import statsmodels.api as sm

# ==== 1. Load data ====
df = pd.read_csv("../data/ec2.csv")
X = df["x"].values
Y1 = df["y1"].values
Y2 = df["y2"].values

# ==== 2. Fit X ~ Normal(μ, σ) ====
mu_X = np.mean(X)
sigma_X = np.std(X, ddof=1)

print("X ~ Normal parameters: mu =", mu_X, ", sigma =", sigma_X)

# ==== 3. Fit factor model Yi = αi + βi X + εi ====


def fit_factor(Y, X):
    X_reg = sm.add_constant(X)
    model = sm.OLS(Y, X_reg).fit()
    alpha, beta = model.params
    residuals = model.resid
    return alpha, beta, residuals


alpha1, beta1, eps1 = fit_factor(Y1, X)
alpha2, beta2, eps2 = fit_factor(Y2, X)

print("Alpha1, Beta1 =", alpha1, beta1)
print("Alpha2, Beta2 =", alpha2, beta2)

# ==== 4. Fit residuals εi ~ T(0, σi, νi) ====

# locate=0, scale estimated σi
nu1, loc1, scale1 = t.fit(eps1, floc=0)
nu2, loc2, scale2 = t.fit(eps2, floc=0)

print("Eps1 T params: nu =", nu1, ", sigma =", scale1)
print("Eps2 T params: nu =", nu2, ", sigma =", scale2)

# ==== 5. Monte-Carlo Simulation (100,000 paths) ====

K = 100_000

# Simulate X
X_sim = np.random.normal(loc=mu_X, scale=sigma_X, size=K)

# Simulate epsilons from fitted t-distributions
eps1_sim = scale1 * np.random.standard_t(df=nu1, size=K)
eps2_sim = scale2 * np.random.standard_t(df=nu2, size=K)

# Compute simulated returns for Y1, Y2
Y1_sim = alpha1 + beta1 * X_sim + eps1_sim
Y2_sim = alpha2 + beta2 * X_sim + eps2_sim

# ==== 6. Convert returns to simulated prices ====

P1_0 = 10  # initial price
P2_0 = 50

P1_sim = P1_0 * (1 + Y1_sim)
P2_sim = P2_0 * (1 + Y2_sim)

# ==== 7. Portfolio P&L ====

PnL = 100 * (P1_sim - P1_0) + 100 * (P2_sim - P2_0)

# ==== 8. Compute 99% VaR and ES ====

alpha = 0.01  # 99% VaR

VaR_99 = -np.percentile(PnL, alpha * 100)
ES_99 = -PnL[PnL <= np.percentile(PnL, alpha * 100)].mean()

print("\n=== Portfolio Risk Metrics (in $ P&L terms) ===")
print("99% VaR ($ loss) =", VaR_99)
print("99% ES ($ loss)  =", ES_99)
