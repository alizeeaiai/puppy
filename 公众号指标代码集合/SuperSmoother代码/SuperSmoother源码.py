from numpy import cos, sin, exp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import gaussian_kde
import numpy as np
import numpy as np
import warnings
import math

warnings.filterwarnings('ignore')

length = 20
alpha1 = 0
HP = 0
a1 = 0
b1 = 0
c1 = 0
c2 = 0
c3 = 0
Filt = 0
HighestC = 0
LowestC = 0
count = 0
Stoc = 0
MESAStochastic = 0

# 1.读取csv数据
df = pd.read_excel('./510050_15.xlsx')

def Cosine(x):
    return cos(x)

def Sine(x):
    return sin(x)

# 2.计算alpha1
alpha1 = (Cosine(.707*2 * math.pi / 48) + Sine (.707 * 2 * math.pi / 48) - 1) / Cosine(.707 * 2 * math.pi / 48)

# 3.计算hp
# 新建一列hp
df['hp'] = 0

for i in range(2, df.shape[0]):
    df['hp'][i] = (1 - alpha1 / 2) * (1 - alpha1 / 2) * (df['close'][i] - 2 * df['close'][i-1] + df['close'][i-2]) + 2 * (1 - alpha1) * df['hp'][i-1] - (1 - alpha1) * (1 - alpha1) * df['hp'][i-2]

# 4.平滑
a1 = np.exp(-1.414 * 3.14159 / 10)
b1 = 2 * a1 * np.cos(1.414 * math.pi / 10)
c2 = b1
c3 = -a1*a1
c1 = 1 - c2 - c3
df['filt'] = 0
for i in range(3, df.shape[0]):
    df['filt'][i] = c1*(df['hp'][i] + df['hp'][i-1]) / 2 + c2 * df['hp'][i-2] + c3 * df['hp'][i-3]

print(df)

# 5.计算HighestC和LowestC
length = 20

HighestC = df['filt']
LowestC = df['filt']
df['HighestC'] = 0
df['LowestC'] = 0
for i in range(df.shape[0]):
    for count in range(length):
        if df['filt'][count] > HighestC[i]:
            df['HighestC'][i] = df['filt'][count]
        if df['filt'][count] < LowestC[i]:
            df['LowestC'][i] = df['filt'][count]
print(df)

# 6.计算最终的mesa指标
df['stoc'] = (df['filt'] - df['LowestC']) / (df['HighestC'] - df['LowestC'])
df['mesa'] = 0
for i in range(3, df.shape[0]):
    df['mesa'][i] = c1*(df['stoc'][i] + df['stoc'][i-1]) / 2 + c2 * df['mesa'][i-1] + c3 * df['mesa'][i-2]
print(df)