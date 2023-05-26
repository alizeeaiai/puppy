import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import os
import warnings

warnings.filterwarnings('ignore')

data_path = '/Users/aiailyu/PycharmProjects/第2节课/'
data_day = pd.read_excel(data_path + '1_510050_d.xlsx')
data_day.rename(columns={'etime':'timestamp'}, inplace=True)
data_day['timestamp'] = pd.to_datetime(data_day['timestamp']).dt.date
data_day = data_day.drop_duplicates('timestamp').reset_index()
data_day = data_day.set_index('timestamp')
del data_day['index']
del data_day['amount']

# 原来的滚动标准化方式
factor_value = pd.DataFrame()
factor_in_use = data_day.copy()
for i in range(factor_in_use.shape[0] + 1):
    factors_mean = factor_in_use.iloc[:i, :].mean(axis=0)
    factors_std = factor_in_use.iloc[:i, :].std(axis=0)
    factor_data = (factor_in_use.iloc[:i, :] - factors_mean) / factors_std
    if i > 0 : factor_value = factor_value.append(factor_data.iloc[-1, :])

# 第一种滚动标准化的方式
actor_value_2 = pd.DataFrame()
factor_in_use_2 = data_day.copy()
test_cumsum = factor_in_use_2.cumsum()
factors_mean_2 = factor_in_use_2.cumsum() / np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
factors_std_2 = factor_in_use_2.expanding().std()
factor_value_2 = (factor_in_use_2-factors_mean_2) / factors_std_2

# 第二种滚动标准化的方式
factor_value_3 = pd.DataFrame()
factor_in_use_3 = data_day.copy()
window = 80
factors_mean_3 = factor_in_use_3.rolling(window=window, min_periods=1).sum() / np.arange(1, factor_in_use_3.shape[0] + 1)[:, np.newaxis]
factors_std_3 = factor_in_use_3.rolling(window=window, min_periods=1).std()
factor_value_3 = (factor_in_use_3 - factors_mean_3) / factors_std_3

