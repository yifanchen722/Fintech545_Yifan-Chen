import numpy as np
from scipy.stats import norm


# ================================
# Black–Scholes Put Delta
# ================================
def put_delta(S, K, T, r, sigma):
    if T <= 0:
        # 到期日 Delta = -1 (in-the-money) 或 0 (OTM)
        return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) - 1  # Put Delta
    return delta


# ================================
# 输入参数（与你之前的题一样）
# ================================
S0 = 100
K = 100
T = 1.0
r = 0.02
sigma = 0.2062  # 用上题算出的 implied volatility

# 计算 Put Delta
delta_put = put_delta(S0, K, T, r, sigma)
print("Put Delta =", delta_put)

# Market maker 卖出期权后的对冲仓位
hedge_shares = -delta_put
print("Shares to buy for hedging =", hedge_shares)

# 回答：For each put option the market maker has sold, she should buy approximately 0.44 shares of the underlying stock to establish a delta-neutral hedge.
