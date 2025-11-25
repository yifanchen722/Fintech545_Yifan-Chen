import numpy as np
import pandas as pd
from scipy.stats import norm
from test_lib import bs_option_greeks

option_type = "Put"
S = -100
K = 90
T = 1 / 255
r = -0.04
q = 0
sigma = 0
position = "Short"

value, delta, gamma, vega, rho, theta = bs_option_greeks(
    option_type, S, K, T, r, q, sigma, position="Short"
)

print("Value:", value)
print("Delta:", delta)
print("Gamma:", gamma)
print("Vega:", vega)
print("Rho:", rho)
print("Theta:", theta)
