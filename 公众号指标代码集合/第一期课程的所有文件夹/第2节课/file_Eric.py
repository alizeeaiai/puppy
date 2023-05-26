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
del data_day['index']
del data_day['amount']
data_day = data_day.set_index('timestamp')

# # 第三部分：数据/因子标准化处理————体会滚动标准化的意义
# factor_in_use = data_day
# factor_value = pd.DataFrame()
# for i in range(factor_in_use.shape[0] + 1):
#     tmp = factor_in_use.iloc[:i, :]
#     factors_mean = factor_in_use.iloc[:i, :].mean(axis=0)
#     factors_std = factor_in_use.iloc[:i, :].std(axis=0)
#     factor_data = (factor_in_use.iloc[:i, :] - factors_mean) / factors_std    # 这是性能灾难，只需要最后一行但是每次都整体计算了i行的df
#     if i > 0 : factor_value = factor_value.append(factor_data.iloc[-1, :])
# print(factor_value)

# Eric 尝试2 (推荐的实现方式) - cumsum() 和 expanding() -
factor_value_2 = pd.DataFrame()
factor_in_use_2 = data_day.copy()
# t_np = np.arange(1, factor_in_use_2.shape[0] + 1)
# print(t_np)
# test_np = np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
# print(test_np)
test_cumsum = factor_in_use_2.cumsum()
factors_mean_2 = factor_in_use_2.cumsum() / np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
factors_std_2 = factor_in_use_2.expanding().std()
factor_value_2 = (factor_in_use_2-factors_mean_2) / factors_std_2
print(factor_value_2)

# # Eric 尝试3 -  指定的窗口rolling, 这个数值是固定窗口内的zscore的值，而不是累积的所有先前的值.
# factor_value_3 = pd.DataFrame()
# factor_in_use_3 = factor_in_use.copy()
# start_3 = time.time()
# window = 80
# factors_mean_3 = factor_in_use_3.rolling(window=window, min_periods=1).sum() / np.arange(1, factor_in_use_3.shape[0] + 1)[:, np.newaxis]
# factors_std_3 = factor_in_use_3.rolling(window=window, min_periods=1).std()
# factor_value_3 = (factor_in_use_3 - factors_mean_3) / factors_std_3
# print("----------Method 3 ------指定的窗口rolling-----{}".format(time.time()-start_3))
# factor_value_3.to_csv('factor_value_3.csv')
#
#
# #
# factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
# factor_value = factor_value.reset_index()
# factor_value.rename(columns={'index':'timestamp'}, inplace=True)
# print(factor_value)
# factor_value.to_csv('factor_value_file_510050.csv')
#
#
#
# #
# #
# # # 第四部分：使用matplotlib画图显示
# # data_15mins_plot = data_15mins[-1000:]
# # data_15mins_plot = data_15mins_plot.reset_index()
# # highs = data_15mins_plot['high']
# # lows = data_15mins_plot['low']
# # fig = plt.figure(figsize=(8,6))
# # ax1 = fig.add_subplot(111, ylabel='stock price')
# # highs.plot(ax=ax1, color='c', lw=2.)
# # lows.plot(ax=ax1, color='y', lw=2.)
# # plt.hlines(highs.head(200).max(),lows.index.values[0],lows.index.values[-1], linewidth=1, color='g')
# # plt.hlines(lows.head(200).min(),lows.index.values[0],lows.index.values[-1], linewidth=1, color='r')
# # plt.axvline(linewidth=2,color='b',x=lows.index.values[200],linestyle=':')
# # plt.legend()
# # plt.grid()
# # plt.show()
#
#
# 第5部分：使用matplotlib画图 - 测试双坐标轴显示

factor_value['timestamp'] = pd.to_datetime(factor_value['timestamp'])
evenly_space = int(factor_value.shape[0]/1000)
factor_value_plot = factor_value.iloc[::evenly_space].sort_values('timestamp')    #等间隔的选出1000多个sample点，不一定是整数

data_15mins = data_15mins.reset_index()
data_15mins.rename(columns={'index':'timestamp'}, inplace=True)
data_15mins['timestamp'] = pd.to_datetime(data_15mins['timestamp'])
factor_value_plot = pd.merge(factor_value_plot, data_15mins, how='left', on='timestamp', suffixes=('_fact', '_raw'))

fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(111, ylabel='raw_price vs. expanding zscore price')

x = factor_value_plot['timestamp']  #如果使用.values, 则会转换为np的datetime64,不是datetime
y1 = factor_value_plot['y_12_fact']
y2 = factor_value_plot['close_raw']
# ax1.plot(x, y1, color='g', linewidth=2, linestyle='-', label='close') # 没搞懂为何加上了x，就不显示线了.
ax1.plot(y1, color='g', linewidth=2, linestyle='-', label='y_12_fact')

ax1.set_xlim(-1, len(x)+1)
# ax1.set_ylim(np.vstack([y1, y2]).min()*0.8, np.vstack([y1, y2]).max()*1.2)

x_tick = range(0, len(x), 50)  #如果x_tick太多放不下的话，是会展现出错的
x_label = [x.iloc[i].strftime('%Y/%m/%d') for i in x_tick]

ax1.set_xticks(x_tick)
ax1.set_xticklabels(x_label, rotation=90)
ax1.legend(loc='upper left', frameon=True)

# 增加第二个坐标系ax2，但是使用与ax1相同的x轴，相当于两个独立的y轴展示
ax2 = ax1.twinx()
# ax2.plot(x, y2, color='r', linewidth=1, linestyle='-', label='close_raw')
ax2.plot(y2, color='r', linewidth=1, linestyle='-', label='close_raw')
ax2.legend(loc='upper right', frameon=True)
plt.grid()
plt.show()