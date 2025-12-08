import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t


# ============================================================
# Global Settings
# ============================================================

RISK_FREE_RATE = 0.02  # Annual risk-free rate for CAPM
TRADING_DAYS_PER_YEAR = 252
EW_COV_LAMBDA = 0.97  # Lambda for EW covariance
EWMA_LAMBDA = 0.94  # Lambda for EWMA mean
BOOTSTRAP_ALPHA = 0.95  # Confidence level for VaR
BOOTSTRAP_HORIZON_DAYS = 252
BOOTSTRAP_SIMS = 10_000
GARCH_HORIZON_DAYS = 252
GARCH_SIMS = 5_000
RANDOM_SEED = 42  # For reproducibility


# ============================================================
# 1. Data
# ============================================================


def load_returns(filepath: str):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df_clean = df.dropna()

    print("Original sample size: ", df.shape)
    print("Cleaned sample size:  ", df_clean.shape)
    print(df_clean.head())

    return df_clean


def compute_portfolio_performance(port_ret: pd.Series):
    ann_ret = port_ret.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan

    return {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
    }


def mvo_weights(mu: pd.Series, sigma: pd.DataFrame):
    sigma_mat = sigma.values
    sigma_inv = np.linalg.inv(sigma_mat)
    mu_vec = mu.values

    w_unscaled = sigma_inv @ mu_vec
    w = w_unscaled / w_unscaled.sum()
    return pd.Series(w, index=mu.index)


# ============================================================
# 2. Covariance: Historical, EW, Near-PSD, Higham
# ============================================================


def ew_covariance(returns: pd.DataFrame, lam: float = EW_COV_LAMBDA):
    r = returns.dropna().values
    T, N = r.shape

    weights = np.array([lam ** (T - 1 - t) for t in range(T)], dtype=float)
    weights = weights / weights.sum()

    mean_w = np.average(r, axis=0, weights=weights)
    X = r - mean_w

    W = np.diag(weights)
    cov_ew = X.T @ W @ X
    return cov_ew


def near_psd(mat: np.ndarray, eps: float = 1e-8):
    vals, vecs = np.linalg.eigh(mat)
    vals_clipped = np.clip(vals, eps, None)
    mat_psd = vecs @ np.diag(vals_clipped) @ vecs.T
    return mat_psd


def higham_near_corr(C: np.ndarray, tol: float = 1e-8, max_iter: int = 100):
    n = C.shape[0]
    Y = C.copy()
    delta_S = np.zeros_like(C)

    for _ in range(max_iter):
        R = Y - delta_S

        # Project onto PSD cone
        vals, vecs = np.linalg.eigh(R)
        vals[vals < 0] = 0.0
        X = vecs @ np.diag(vals) @ vecs.T

        delta_S = X - R
        Y = X.copy()

        # Project onto correlation matrices (unit diagonal)
        np.fill_diagonal(Y, 1.0)

        if np.linalg.norm(Y - C, ord="fro") < tol:
            break

    return Y


def higham_covariance(sigma_in: np.ndarray):
    std = np.sqrt(np.diag(sigma_in))
    D_inv = np.diag(1.0 / std)
    C = D_inv @ sigma_in @ D_inv

    C_high = higham_near_corr(C)
    sigma_high = np.diag(std) @ C_high @ np.diag(std)
    return sigma_high


# ============================================================
# 3. CAPM Expected Returns
# ============================================================


def estimate_capm_mu(df_returns: pd.DataFrame, rf: float = RISK_FREE_RATE):
    start_date = df_returns.index.min().strftime("%Y-%m-%d")
    end_date = df_returns.index.max().strftime("%Y-%m-%d")

    market = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
    market_ret = market["Close"].pct_change()

    if isinstance(market_ret, pd.DataFrame):
        if "Close" in market_ret.columns:
            market_ret_df = market_ret.rename(columns={"Close": "MKT_ret"})
        else:
            col0 = market_ret.columns[0]
            market_ret_df = market_ret.rename(columns={col0: "MKT_ret"})
    else:
        market_ret_df = market_ret.to_frame(name="MKT_ret")

    df_capm = df_returns.join(market_ret_df, how="inner").dropna()
    print("\nCAPM regression sample size: ", df_capm.shape)
    print(df_capm.head())

    assets = df_returns.columns.tolist()
    betas = {}

    for asset in assets:
        y = df_capm[asset]
        X = sm.add_constant(df_capm["MKT_ret"])
        model = sm.OLS(y, X).fit()
        betas[asset] = model.params["MKT_ret"]

    print("\nCAPM betas:")
    print(betas)

    mu_mkt = df_capm["MKT_ret"].mean() * TRADING_DAYS_PER_YEAR

    mu_capm = {asset: rf + betas[asset] * (mu_mkt - rf) for asset in assets}

    mu_capm = pd.Series(mu_capm)
    print("\nCAPM expected returns (annualized):")
    print(mu_capm)

    return mu_capm


# ============================================================
# 4. EWMA Expected Returns
# ============================================================


def ewma_expected_returns(df_returns: pd.DataFrame, lam: float = EWMA_LAMBDA):
    alpha = 1 - lam
    ewma_daily_mean = df_returns.ewm(alpha=alpha, adjust=False).mean()
    mu_daily = ewma_daily_mean.iloc[-1]
    mu_annual = mu_daily * TRADING_DAYS_PER_YEAR

    print("\nEWMA expected returns (annualized):")
    print(mu_annual)

    return mu_annual


# ============================================================
# 5. Risk Measures: Bootstrap VaR/CVaR & GARCH VaR/CVaR
# ============================================================


def bootstrap_var_cvar(
    port_ret: pd.Series,
    horizon_days: int = BOOTSTRAP_HORIZON_DAYS,
    n_sims: int = BOOTSTRAP_SIMS,
    alpha: float = BOOTSTRAP_ALPHA,
):
    ret_arr = port_ret.dropna().values
    sims = []

    for _ in range(n_sims):
        sample = np.random.choice(ret_arr, size=horizon_days, replace=True)
        annual_ret = sample.sum()
        sims.append(annual_ret)

    sims = np.array(sims)
    losses = -sims

    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()

    return var, cvar


def garch_simulate_paths(
    port_ret: pd.Series,
    horizon_days: int = GARCH_HORIZON_DAYS,
    n_sims: int = GARCH_SIMS,
    alpha: float = BOOTSTRAP_ALPHA,
):
    r = port_ret.dropna() * 100  # scale to percent for stability

    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
    res = am.fit(disp="off")

    omega = res.params["omega"]
    alpha1 = res.params["alpha[1]"]
    beta1 = res.params["beta[1]"]
    mu = res.params["mu"]

    last_var = res.conditional_volatility[-1] ** 2

    sims = []

    for _ in range(n_sims):
        var_t = last_var
        path_returns = []

        for _ in range(horizon_days):
            z = np.random.randn()
            ret_pct = mu + np.sqrt(var_t) * z
            path_returns.append(ret_pct / 100.0)

            var_t = omega + alpha1 * (ret_pct - mu) ** 2 + beta1 * var_t

        annual_ret = np.sum(path_returns)
        sims.append(annual_ret)

    sims = np.array(sims)
    losses = -sims

    VaR = np.quantile(losses, alpha)
    CVaR = losses[losses >= VaR].mean()

    return VaR, CVaR


# ============================================================
# 6. Distribution-Based VaR / ES (Normal vs t)
# ============================================================


def analytical_var_normal(r: pd.Series, alpha: float = 0.95):
    r = r.dropna()
    mu = r.mean()
    sigma = r.std()

    q_ret = norm.ppf(1 - alpha, loc=mu, scale=sigma)
    var = -q_ret
    return var, mu, sigma


def analytical_var_t(r: pd.Series, alpha: float = 0.95):
    r = r.dropna()
    df_hat, loc_hat, scale_hat = t.fit(r)
    q_ret = t.ppf(1 - alpha, df_hat, loc=loc_hat, scale=scale_hat)
    var = -q_ret
    return var, df_hat, loc_hat, scale_hat


def simulate_var_es_from_dist(
    dist, params: tuple, n_sims: int = 100_000, alpha: float = 0.95
):
    if dist is norm:
        mu, sigma = params
        sims = norm.rvs(loc=mu, scale=sigma, size=n_sims)
    else:
        df_hat, loc_hat, scale_hat = params
        sims = t.rvs(df_hat, loc=loc_hat, scale=scale_hat, size=n_sims)

    losses = -sims
    var_sim = np.quantile(losses, alpha)
    es_sim = losses[losses >= var_sim].mean()
    return var_sim, es_sim


# ============================================================
# 7. Plotting
# ============================================================


def plot_cumulative_returns(portfolios: dict[str, pd.Series]):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))

    for name, series in portfolios.items():
        plt.plot(series.cumsum(), label=name)

    plt.title("Cumulative Portfolio Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weight_comparison(weights_dict: dict[str, pd.Series]):
    weights_df = pd.DataFrame(weights_dict)
    weights_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Optimal Portfolio Weights")
    plt.tight_layout()
    plt.show()


def plot_return_distributions(portfolios: dict[str, pd.Series]):
    plt.figure(figsize=(10, 5))
    for name, series in portfolios.items():
        sns.histplot(series.dropna(), kde=True, label=name, stat="density", alpha=0.4)

    plt.title("Distribution of Daily Portfolio Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Main Pipeline
# ============================================================


def main():
    np.random.seed(RANDOM_SEED)

    # Load Data
    df_clean = load_returns("combined_returns_model_option.csv")
    assets = df_clean.columns.tolist()
    print("Asset list:", assets)

    # Model 1: Historical Mean + MVO
    mu_hist = df_clean.mean() * TRADING_DAYS_PER_YEAR
    sigma_hist = df_clean.cov() * TRADING_DAYS_PER_YEAR

    print("\nHistorical mean (annualized):")
    print(mu_hist)
    print("\nHistorical covariance (annualized):")
    print(sigma_hist)

    weights_hist = mvo_weights(mu_hist, sigma_hist)
    print("\nOptimal weights under Historical Mean model:")
    print(weights_hist)
    print("Sum of weights:", weights_hist.sum())

    port_ret_hist = (df_clean * weights_hist).sum(axis=1)
    perf_hist = compute_portfolio_performance(port_ret_hist)
    print("\nHistorical Mean portfolio performance:")
    print(perf_hist)

    # Enhanced Covariance: EW + PSD / Higham
    sigma_ew_daily = ew_covariance(df_clean, lam=EW_COV_LAMBDA)
    sigma_ew = sigma_ew_daily * TRADING_DAYS_PER_YEAR
    sigma_ew_df = pd.DataFrame(sigma_ew, index=assets, columns=assets)

    print("\nEW covariance (annualized, lambda=0.97):")
    print(sigma_ew_df)

    # EW covariance + Historical mean
    weights_hist_ew = mvo_weights(mu_hist, sigma_ew_df)
    port_ret_hist_ew = (df_clean * weights_hist_ew).sum(axis=1)
    perf_hist_ew = compute_portfolio_performance(port_ret_hist_ew)

    print("\nHistorical mean + EW covariance portfolio performance:")
    print(perf_hist_ew)

    # EW covariance + near-PSD
    sigma_ew_psd = near_psd(sigma_ew)
    sigma_ew_psd_df = pd.DataFrame(sigma_ew_psd, index=assets, columns=assets)
    weights_hist_ew_psd = mvo_weights(mu_hist, sigma_ew_psd_df)
    port_ret_hist_ew_psd = (df_clean * weights_hist_ew_psd).sum(axis=1)
    perf_hist_ew_psd = compute_portfolio_performance(port_ret_hist_ew_psd)

    print("\nEW covariance + near-PSD portfolio performance:")
    print(perf_hist_ew_psd)

    # EW covariance + Higham
    sigma_ew_high = higham_covariance(sigma_ew)
    sigma_ew_high_df = pd.DataFrame(sigma_ew_high, index=assets, columns=assets)
    weights_hist_ew_high = mvo_weights(mu_hist, sigma_ew_high_df)
    port_ret_hist_ew_high = (df_clean * weights_hist_ew_high).sum(axis=1)
    perf_hist_ew_high = compute_portfolio_performance(port_ret_hist_ew_high)

    print("\nEW covariance + Higham portfolio performance:")
    print(perf_hist_ew_high)

    # Model 2: CAPM Expected Returns + MVO
    mu_capm = estimate_capm_mu(df_clean, rf=RISK_FREE_RATE)
    weights_capm = mvo_weights(mu_capm, sigma_hist)
    port_ret_capm = (df_clean * weights_capm).sum(axis=1)
    perf_capm = compute_portfolio_performance(port_ret_capm)

    print("\nCAPM portfolio weights:")
    print(weights_capm)
    print("\nCAPM portfolio performance:")
    print(perf_capm)

    # Model 3: EWMA Expected Returns + MVO
    mu_ewma = ewma_expected_returns(df_clean, lam=EWMA_LAMBDA)
    weights_ewma = mvo_weights(mu_ewma, sigma_hist)
    port_ret_ewma = (df_clean * weights_ewma).sum(axis=1)
    perf_ewma = compute_portfolio_performance(port_ret_ewma)

    print("\nEWMA portfolio weights:")
    print(weights_ewma)
    print("\nEWMA portfolio performance:")
    print(perf_ewma)

    # Bootstrap VaR / CVaR
    var_hist, cvar_hist = bootstrap_var_cvar(port_ret_hist)
    var_capm, cvar_capm = bootstrap_var_cvar(port_ret_capm)
    var_ewma, cvar_ewma = bootstrap_var_cvar(port_ret_ewma)

    print("\n===== Bootstrap VaR / CVaR (1-year, 95%) =====")
    print(f"Historical Mean: VaR = {var_hist:.3f}, CVaR = {cvar_hist:.3f}")
    print(f"CAPM:            VaR = {var_capm:.3f}, CVaR = {cvar_capm:.3f}")
    print(f"EWMA:            VaR = {var_ewma:.3f}, CVaR = {cvar_ewma:.3f}")

    # GARCH(1,1) VaR / CVaR
    garch_var_hist, garch_cvar_hist = garch_simulate_paths(port_ret_hist)
    garch_var_capm, garch_cvar_capm = garch_simulate_paths(port_ret_capm)
    garch_var_ewma, garch_cvar_ewma = garch_simulate_paths(port_ret_ewma)

    print("\n===== GARCH(1,1) VaR / CVaR (1-year, 95%) =====")
    print(f"Historical Mean: VaR = {garch_var_hist:.4f}, CVaR = {garch_cvar_hist:.4f}")
    print(f"CAPM:            VaR = {garch_var_capm:.4f}, CVaR = {garch_cvar_capm:.4f}")
    print(f"EWMA:            VaR = {garch_var_ewma:.4f}, CVaR = {garch_cvar_ewma:.4f}")

    # Distribution-based VaR / ES (Normal vs t)
    alpha = BOOTSTRAP_ALPHA

    for name, r in [
        ("Historical Mean", port_ret_hist),
        ("CAPM", port_ret_capm),
        ("EWMA", port_ret_ewma),
    ]:
        r_clean = r.dropna()
        var_norm_ana, mu_norm, sigma_norm = analytical_var_normal(r_clean, alpha=alpha)
        var_t_ana, df_t, loc_t, scale_t = analytical_var_t(r_clean, alpha=alpha)

        var_norm_sim, es_norm_sim = simulate_var_es_from_dist(
            norm, (mu_norm, sigma_norm), alpha=alpha
        )
        var_t_sim, es_t_sim = simulate_var_es_from_dist(
            t, (df_t, loc_t, scale_t), alpha=alpha
        )

        print(f"\n===== {name} Portfolio: Normal vs t VaR/ES (1-day, 95%) =====")
        print(f"Analytical Normal VaR: {var_norm_ana:.6f}")
        print(f"Simulated  Normal VaR: {var_norm_sim:.6f}, ES: {es_norm_sim:.6f}")
        print(f"Analytical t VaR:      {var_t_ana:.6f}")
        print(f"Simulated  t VaR:      {var_t_sim:.6f}, ES: {es_t_sim:.6f}")

    # Plots
    portfolios = {
        "Historical Mean": port_ret_hist,
        "CAPM": port_ret_capm,
        "EWMA": port_ret_ewma,
    }
    plot_cumulative_returns(portfolios)

    weights_dict = {
        "Historical Mean": weights_hist,
        "CAPM": weights_capm,
        "EWMA": weights_ewma,
    }
    plot_weight_comparison(weights_dict)

    plot_return_distributions(portfolios)


if __name__ == "__main__":
    main()
