import pandas as pd
import talib

df = pd.read_excel('./510050_15.xlsx')
df['20ma'] = df['close'].rolling(20).mean()
df['slope'] = talib.LINEARREG_SLOPE(df['20ma'],5)
print(df)