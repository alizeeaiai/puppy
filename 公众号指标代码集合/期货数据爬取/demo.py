import pandas as pd

data = pd.read_excel('./510050_d.xlsx')
print(data['high'])
rolling_mean = data['high'].expanding().mean()
print(rolling_mean)