import numpy as np
import pandas as pd
file_path = './fct_data.csv'
df = pd.read_csv(file_path)
df = df.drop(columns='timestamp', axis=1)
df['noise_0'] = (np.random.randn(df.shape[0]) - 0.5) / 10
df['noise_1'] = 0.0
df['noise_2'] = 1.0
df['noise_3'] = -1.0
print(df.corr())

