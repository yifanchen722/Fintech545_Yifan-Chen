import numpy as np
import pandas as pd
from scipy.stats import norm


# Identify GBSM
def bs_option_greeks(option_type, S, K, T, r, q, sigma):

    # Calculate d1, d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    value = 0.0
    delta = 0.0
    gamma = 0.0
    vega = 0.0
    rho = 0.0
    theta = 0.0

    # Calculate option price
    if option_type == "Call":
        value = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        value = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    # calculate Greeks
    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    if option_type == "Call":
        theta = (
            -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
    elif option_type == "Put":
        theta = (
            -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
    return value, delta, gamma, vega, rho, theta


path = "../final/p3.csv"
data = pd.read_csv(path)

results = []
for _, row in data.iterrows():
    T = row["DaysToMaturity"] / row["DayPerYear"]
    value, delta, gamma, vega, rho, theta = bs_option_greeks(
        row["Option Type"],
        row["Underlying"],
        row["Strike"],
        T,
        row["RiskFreeRate"],
        row["DividendRate"],
        row["ImpliedVol"],
    )
    results.append([row["ID"], value, delta, gamma, vega, rho, theta])

print(results)

df_result = pd.DataFrame(
    results, columns=["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]
)
print(df_result.iloc[0:-1, :])
