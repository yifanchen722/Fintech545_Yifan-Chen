import pandas as pd
import numpy as np


def american_option_discrete_dividends(
    S, K, T, r, sigma, option_type, dividend_dates, dividend_amts, N=200
):

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    if not dividend_dates:
        return american_option_binomial(S, K, T, r, sigma, option_type, N)

    first_div_time = dividend_dates[0]
    first_div_amt = dividend_amts[0]

    if first_div_time >= T:
        return american_option_binomial(S, K, T, r, sigma, option_type, N)

    step_div = int(np.floor(first_div_time / dt))
    if step_div < 1:
        step_div = 1

    ST = np.zeros((step_div + 1, step_div + 1))
    for i in range(step_div + 1):
        ST[step_div, i] = S * (u**i) * (d ** (step_div - i))

    for n in range(step_div - 1, -1, -1):
        for i in range(n + 1):
            ST[n, i] = S * (u**i) * (d ** (n - i))

    V = np.zeros(step_div + 1)
    for i in range(step_div + 1):
        S_after_div = max(ST[step_div, i] - first_div_amt, 0)

        sub_value = american_option_discrete_dividends(
            S_after_div,
            K,
            T - first_div_time,
            r,
            sigma,
            option_type,
            dividend_dates[1:],
            dividend_amts[1:],
            N,
        )
        V[i] = sub_value

    for n in range(step_div - 1, -1, -1):
        for i in range(n + 1):
            St = S * (u**i) * (d ** (n - i))
            cont = np.exp(-r * dt) * (p * V[i + 1] + (1 - p) * V[i])
            if option_type.lower() == "call":
                exercise = max(St - K, 0)
            else:
                exercise = max(K - St, 0)
            V[i] = max(cont, exercise)

    return V[0]


def american_option_binomial(S, K, T, r, sigma, option_type, N=200):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    ST = S * d ** np.arange(N, -1, -1) * u ** np.arange(0, N + 1, 1)
    if option_type.lower() == "call":
        V = np.maximum(ST - K, 0)
    else:
        V = np.maximum(K - ST, 0)

    for n in range(N - 1, -1, -1):
        ST = ST[1:] / u
        V = np.exp(-r * dt) * (p * V[1:] + (1 - p) * V[:-1])
        if option_type.lower() == "call":
            V = np.maximum(V, ST - K)
        else:
            V = np.maximum(V, K - ST)
    return V[0]


path = "../data/test12_3.csv"
data = pd.read_csv(path)
results = []

for idx, row in data.iterrows():
    dividend_dates = [
        float(x) / row["DayPerYear"] for x in str(row["DividendDates"]).split(",")
    ]
    dividend_amts = [float(x) for x in str(row["DividendAmts"]).split(",")]

    value = american_option_discrete_dividends(
        S=row["Underlying"],
        K=row["Strike"],
        T=row["DaysToMaturity"] / row["DayPerYear"],
        r=row["RiskFreeRate"],
        sigma=row["ImpliedVol"],
        option_type=row["Option Type"],
        dividend_dates=dividend_dates,
        dividend_amts=dividend_amts,
        N=200,
    )
    results.append({"ID": row["ID"], "Value": value})

result_df = pd.DataFrame(results)
print(result_df)
