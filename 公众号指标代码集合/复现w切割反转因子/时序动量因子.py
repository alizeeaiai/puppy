# step_1 当前bar涨跌幅除以amount者volume，获得当天amount或者volume一单位驱动价格上涨/下跌的力量，记为A；
# step_2 把A标准化，然后统计过去20天之内的涨跌幅，并分别统计这20天推动上涨/下跌的A的总和，分别记作rise_A和fall_A
# step_3 求出rise_A和fall_A的标准差，用这个标准差处理过去20天的涨跌幅，代表时序推动力度B；
# step_4 把B和return之间做回归，计算因子B的sharpe

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 数据预处理
df = pd.read_csv('./股票数据大集合/宁德时代每日成交金额.csv')
df = df.drop(['Unnamed: 0', 'ts_code', 'open', 'high', 'low', 'amount'], axis=1)
df.reset_index(drop=True, inplace=True)

# step_1 当前bar涨跌幅除以amount者volume，获得当天amount或者volume一单位驱动价格上涨/下跌的力量，记为A；
df['A'] = df['pct_chg'] / df['vol']

# step_2 把A标准化，然后统计过去20天之内的涨跌幅，并分别统计这20天推动上涨/下跌的A的总和，分别记作rise_A和fall_A
mean = np.mean(df['A'])
std = np.std(df['A'])
df['A'] = (df['A'] - mean) / std
# 统计这20天的涨跌幅
df['rolling_sum'] = df['pct_chg'].rolling(window=20).sum()
# 新建一列，pct_chg为正，取值A。否则为0
def get_rise_a(row):
    return row['A'] if row['pct_chg'] > 0 else 0
def get_fall_a(row):
    return row['A'] if row['pct_chg'] < 0 else 0
# 使用apply()方法调用函数，将结果存储在新的rise_a列中
df['rise_a'] = df.apply(get_rise_a, axis=1).rolling(window=20).sum()
df['fall_a'] = df.apply(get_fall_a, axis=1).rolling(window=20).sum()
# step_3 求出rise_A和fall_A的标准差，用这个标准差乘以20天的涨跌幅，代表时序推动力度B；
df['rise_a'] = df['rise_a'].rolling(window=20).std()
df['fall_a'] = df['fall_a'].rolling(window=20).std()

# step_4 把B和return之间做回归，计算因子B的sharpe
