import test_lib as tl

path = "../data/test2.csv"
lam = 0.94

corr = tl.get_ew_corr(path, lam, has_date=False, is_rate=True)
print(corr)
