import pandas as pd

df = pd.read_csv('leagues.csv', sep=";", header=0)
print(df)