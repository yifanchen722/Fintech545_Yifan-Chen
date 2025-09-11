import csv
from decimal import Decimal, getcontext, ROUND_HALF_UP

getcontext().prec = 80

file_path = "/Users/nico/Desktop/test7_1.csv"
col_name = "x1"

values = []
with open(file_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        values.append(Decimal(row[col_name]))

n = Decimal(len(values))
s = sum(values)
mu = s / n
ss = sum((v - mu) ** 2 for v in values)
sigma = (ss / (n - Decimal("1"))).sqrt() if n > 1 else Decimal("0")

mu_q = mu.quantize(Decimal("1e-18"), rounding=ROUND_HALF_UP)
sigma_q = sigma.quantize(Decimal("1e-17"), rounding=ROUND_HALF_UP)

mu_str = format(mu_q, "f")
sigma_str = format(sigma_q, "f")

print("mu:", mu_str)
print("sigma:", sigma_str)

with open("/Users/nico/Desktop/testout.csv", "w", encoding="utf-8") as f:
    f.write(f"{mu_str}\t{sigma_str}\n")
