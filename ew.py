import numpy as np
import pandas as pd
from scipy.stats import norm
import test_lib

path = "../final/problem3_insample.csv"
lam = 0.97

cov = test_lib.get_ew_cov(path, lam, has_date=True, is_rate=True)
print(cov)
