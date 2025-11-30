import pandas as pd

real = pd.read_csv("real.csv")
synthetic = pd.read_csv("synthetic.csv")

print("REAL columns:", real.columns.tolist())
print("SYNTHETIC columns:", synthetic.columns.tolist())
