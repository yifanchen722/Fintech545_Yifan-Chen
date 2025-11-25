import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_lib import bs_option_greeks

df = pd.read_csv("../data/question3.csv")
S_5days = df["fwdPrices"].values

r = 0.04
sigma = 0.20
K = 90
q = 0
S0 = 100

T0 = 30 / 365
T5 = 25 / 365

# Initial put premium (Short)
P0, *_ = bs_option_greeks("Put", S0, K, T0, r, q, sigma)

# Compute P&L
P_5days = np.array([bs_option_greeks("Put", S, K, T5, r, q, sigma)[0] for S in S_5days])

# short put
PnL = P0 - P_5days

# if long put
# PnL = P_5days - P0


# Plot P&L
plt.figure(figsize=(8, 5))
plt.scatter(S_5days, PnL, alpha=0.4)
plt.axhline(0, color="black")
plt.xlabel("Underlying Price after 5 days")
plt.ylabel("P&L of Short Put")
plt.title("Short Put P&L after 5 Days")
plt.grid(True)
plt.show()

# VaR & ES
alpha = 0.05  # 95% VaR
sorted_pnl = np.sort(PnL)

VaR = -np.percentile(PnL, 100 * alpha)
ES = -np.mean(sorted_pnl[: int(len(PnL) * alpha)])

print("Initial Short Put Value =", P0)
print("VaR(95%) =", VaR)
print("ES(95%) =", ES)
