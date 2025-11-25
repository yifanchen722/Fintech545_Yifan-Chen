import pandas as pd
import test_lib


df = pd.read_csv("../data/problem2.csv")
x = df["X"]

print("t_distribution")
test_lib.fit_t_distribution(x)

print("normal_distribution")
test_lib.fit_normal_distribution(x)

print("summary_statistics")
test_lib.summ_stats(x)
