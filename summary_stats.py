import pandas as pd
import test_lib as lib

data = pd.read_csv("../data/question2.csv")

lib.summ_stats(data)
