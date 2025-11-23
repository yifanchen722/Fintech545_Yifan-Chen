import numpy as np
import pandas as pd


def american_option_binomial(S, K, T, r, q, sigma, steps, option_type, b=None):
    if steps <= 0:
        raise ValueError("steps must be positive integer")
    dt = T / steps
    if dt <= 0:
        return max(0.0, S - K) if option_type == "Call" else max(0.0, K - S)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    if b is None:
        b_use = r - q
    else:
        b_use = b

    p = (np.exp(b_use * dt) - d) / (u - d)
    p = np.minimum(1.0, np.maximum(0.0, p))

    ST = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])
    if option_type == "Call":
        option_values = np.maximum(ST - K, 0.0)
    else:
        option_values = np.maximum(K - ST, 0.0)

    discount = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        ST_i = np.array([S * (u**j) * (d ** (i - j)) for j in range(i + 1)])
        cont = discount * (
            p * option_values[1 : i + 2] + (1 - p) * option_values[0 : i + 1]
        )
        if option_type == "Call":
            exercise = np.maximum(ST_i - K, 0.0)
        else:
            exercise = np.maximum(K - ST_i, 0.0)
        option_values = np.maximum(cont, exercise)

    return float(option_values[0])


def compute_greeks_with_b(S, K, T, r, q, sigma, steps, option_type):
    eps_s = 0.01 * S
    eps_s_gamma = 0.02 * S
    eps_sigma = 1e-4
    eps_rho = 1e-4
    eps_t = 1.0 / 365.0

    if T <= eps_t:
        eps_t = T / 2.0 if T > 0 else 0.0

    V = american_option_binomial(S, K, T, r, q, sigma, steps, option_type, b=(r - q))

    # Delta
    V_up = american_option_binomial(
        S + eps_s, K, T, r, q, sigma, steps, option_type, b=(r - q)
    )
    V_down = american_option_binomial(
        S - eps_s, K, T, r, q, sigma, steps, option_type, b=(r - q)
    )
    delta = (V_up - V_down) / (2.0 * eps_s)

    # Gamma
    V_up_g = american_option_binomial(
        S + eps_s_gamma, K, T, r, q, sigma, steps, option_type, b=(r - q)
    )
    V_down_g = american_option_binomial(
        S - eps_s_gamma, K, T, r, q, sigma, steps, option_type, b=(r - q)
    )
    gamma = (V_up_g - 2.0 * V + V_down_g) / (eps_s_gamma**2)

    # Vega
    V_sigma_up = american_option_binomial(
        S, K, T, r, q, sigma + eps_sigma, steps, option_type, b=(r - q)
    )
    V_sigma_down = american_option_binomial(
        S, K, T, r, q, sigma - eps_sigma, steps, option_type, b=(r - q)
    )
    vega = (V_sigma_up - V_sigma_down) / (2.0 * eps_sigma)

    # Rho
    eps_rho = 1e-4

    V_r_up = american_option_binomial(
        S, K, T, r + eps_rho, q, sigma, steps, option_type, b=(r - q)
    )
    V_r_down = american_option_binomial(
        S, K, T, max(r - eps_rho, 0), q, sigma, steps, option_type, b=(r - q)
    )

    rho = (V_r_up - V_r_down) / (2 * eps_rho)

    # Theta
    if eps_t > 0:
        V_t_forward = american_option_binomial(
            S, K, max(T - eps_t, 0.0), r, q, sigma, steps, option_type, b=(r - q)
        )
        theta = (V_t_forward - V) / (-eps_t)
    else:
        theta = 0.0

    return V, delta, gamma, vega, rho, theta


path = "../data/test12_1.csv"
data = pd.read_csv(path)

if data.tail(1).isnull().all(axis=1).values[0]:
    data = data.iloc[:-1]

results = []
steps = 300

for _, row in data.iterrows():
    T = row["DaysToMaturity"] / row["DayPerYear"]
    V, delta, gamma, vega, rho, theta = compute_greeks_with_b(
        S=row["Underlying"],
        K=row["Strike"],
        T=T,
        r=row["RiskFreeRate"],
        q=row["DividendRate"],
        sigma=row["ImpliedVol"],
        steps=steps,
        option_type=row["Option Type"],
    )
    results.append([row["ID"], V, delta, gamma, vega, rho, theta])

result_df = pd.DataFrame(
    results,
    columns=["ID", "Value", "Delta", "Gamma", "Vega", "rho", "Theta"],
)

pd.set_option("display.float_format", "{:.10f}".format)
result_df["ID"] = result_df["ID"].astype(int)

print(result_df)
