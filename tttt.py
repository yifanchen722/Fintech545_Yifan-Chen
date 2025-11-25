import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import brentq


K_B = 100
T_B = 10 / 255
C0_B = 7.69
S_B0 = 655
rf = 0.04


def bs_put(S, K, r, sigma, T, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


def put_error(sigma):
    return bs_put(C0_B) - (S_B0, K_B, rf, sigma, T_B)


sigma_B = brentq(lambda x: put_error(x), 1e-6, 3.0)

print("Implied vol for Asset B =", sigma_B)
