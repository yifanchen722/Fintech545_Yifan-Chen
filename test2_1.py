import numpy as np
import pandas as pd
from scipy.stats import norm
import test_lib

path = "../data/quiz2.csv"
lam = 0.94

cov = test_lib.get_ew_cov(path, lam, has_date=True, is_rate=False)
print(cov)
