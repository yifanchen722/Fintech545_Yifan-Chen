import numpy as np
import pandas as pd
import test_lib

# 判断是否为psd
data_path = "../data/testout_3.1.csv"
is_psd = test_lib.tell_psd(data_path)
print(is_psd)

# 计算Cholesky分解
df = pd.read_csv(data_path, sep=",")
data = df.values.astype(float)

L = test_lib.chol_psd(data)
print(L)
