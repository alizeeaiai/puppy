import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import os

# 第一部分 读取本地数据并处理成为feed_data

# 1、读取csv文件
start_time = time.time()
# data_path = 'D:/9_quant_course/' # 将源文件夹地址命名为data_path
HEAD_PATH = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '../..')))    #不放在本project目录下是因为很大量文件的话会让IDE的索引过大
data_path = os.path.join(HEAD_PATH, 'data', 'raw_data')

# data_15mins = pd.read_csv(data_path + '510050.SH_15.csv') # 读取csv文件
raw_file_510050 = os.path.join(data_path, '510050.SH_15.csv')
data_15mins = pd.read_csv(raw_file_510050)  # 读取csv文件
print(data_15mins)

# 2、替换列名
data_15mins.rename(columns={'etime':'timestamp'}, inplace=True)  # 对其中的列进行重新命名
print(type(data_15mins['timestamp'][0]))
data_15mins['timestamp'] = pd.to_datetime(data_15mins['timestamp'])
print(type(data_15mins['timestamp'][0]))

# 3、设置timestamp为索引名称
data_15mins = data_15mins.set_index('timestamp') # 设置timestamp为索引

# 4、loc和iloc提取数据
print(data_15mins.loc['2022-11-29']) # 提取出2022年11月29日的数据
print(data_15mins.iloc[-1000:]) # 提取出最后1000行的数据
# loc和iloc提取列数据
print(data_15mins.loc[:, ['open', 'close']]) # 提取open和close两列数据
print(data_15mins.iloc[:10, 0:2]) # 一定要注意iloc在使用的时候，是不包括冒号后面的数据的，截止他前面的数字，这个在数据滚动标准化方面有大用！


# 5、尝试csv，Excel，pickle三种文件格式的读取速度

# data_15mins.to_excel(data_path + '510050.SH_15.xlsx') # 将文件保存为Excel格式文件
# data_15mins.to_pickle(data_path + '510050.SH_15.pkl') # 将文件保存为Excel格式文件
start_time = time.time()
data_15mins_csv = pd.read_csv(raw_file_510050)
end_time_0 = time.time()
print('================================到此使用时间为： ', end_time_0 - start_time)
# data_15mins_excel = pd.read_excel(data_path + '510050.SH_15.xlsx')
# end_time_1 = time.time()
# print('================================到此使用时间为： ', end_time_1 - end_time_0)
# data_15mins_pkl = pd.read_pickle(data_path + '510050.SH_15.pkl')
# end_time_2 = time.time()
# print('================================到此使用时间为： ', end_time_2 - end_time_1)

# 6、将未来t期收益率计算出来并写入文件

t_delay = [1, 3, 5, 7, 9, 12]
for t in t_delay: # 分别获得未来1,3,5,7,9,12个周期的收益率，并将其shift，作为target
    data_15mins['y_{}'.format(t)] = data_15mins['close'].shift(-t) / data_15mins['close'] - 1
# 注意：需要理解format的用法，注意shift的用法，需要注意数据中出现的nan，inf，-inf等怎么处理？
# print(data_15mins)

# 7、replace函数处理nan，inf，-inf
data_15mins = data_15mins.replace([np.nan, np.inf, -np.inf], 0.0) # 替换掉所有的nan并撤销index
# print(data_15mins)

# 第二部分 本地数据get以后，怎样获得实时数据更新本地数据库？

# 1、调用pytdx库
import pytdx
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API

# 2、处理好data数据，便于后续使用
data = pd.read_csv(raw_file_510050)
data.rename(columns={'etime':'timestamp'}, inplace=True)  # 对其中的列进行重新命名
data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
print(type(data['timestamp'][0])) # 注意此处为string数据类型
data['timestamp'] = pd.to_datetime(data['timestamp'])
print(type(data['timestamp'][0])) # 此处已经处理为timestamp的数据类型
print(data)

# 3、命名current_data作为新get到的数据作为dataframe，后续和data合并
# https://gitee.com/better319/pytdx/
api = TdxHq_API()
if api.connect('119.147.212.81', 7709): # 注意这里的IP地址和数据接口
    current_data = api.to_df(api.get_security_bars(1, 1, '510050', 0, 500)) # 注意这里，第一个1表示是15分钟的数据，其中0为5分钟K线 1 15分钟K线 2 30分钟K线 3 1小时K线 4 日K线
    api.disconnect() # 调用完以后一定要关闭接口
# print(current_data)

# 4、提取current_data的数据
current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
print(current_data)

# 5、核心部分：把两个部分的数据合并在一起
data = pd.concat([data, current_data], axis=0) # 合并数据
print(data)
data = data.sort_values(by='timestamp', ascending=True) # 注意这一步是非常必要的，要以timestamp作为排序基准
print(data)
data = data.drop_duplicates('timestamp').reset_index() # 注意这一步非常重要，以timestamp为基准进行去重处理
print(data)

del data['index']
data = data.set_index('timestamp')
print(data)
data.to_csv(raw_file_510050) # 最终将数据保存


# 第三部分：数据/因子标准化处理————体会滚动标准化的意义
factor_in_use = data_15mins

start_1 = time.time()
factor_value = pd.DataFrame()
for i in range(factor_in_use.shape[0] + 1): # factor_in_use.shape[0]
    tmp = factor_in_use.iloc[:i, :]
    factors_mean = factor_in_use.iloc[ :i, : ].mean(axis=0) # 原因是这里加上了iloc之后，只能计算到i的前一个，
    factors_std = factor_in_use.iloc[ :i, : ].std(axis=0)
    factor_data = (factor_in_use.iloc[ :i, : ] - factors_mean) / factors_std    # 这是性能灾难，只需要最后一行但是每次都整体计算了i行的df
    if i > 0 : factor_value = factor_value.append(factor_data.iloc[-1, :])
print("----------Method 1 ------for loop-----{}".format(time.time()-start_1))
factor_value.to_csv('factor_value.csv')


# 尝试2 (推荐的实现方式) - cumsum() 和 expanding() -
factor_value_2 = pd.DataFrame()
factor_in_use_2 = factor_in_use.copy()

start_2 = time.time()
# t_np = np.arange(1, factor_in_use_2.shape[0] + 1)
# test_np = np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
# test_cumsum = factor_in_use_2.cumsum()
factors_mean_2 = factor_in_use_2.cumsum() / np.arange(1, factor_in_use_2.shape[0] + 1)[:, np.newaxis]
factors_std_2 = factor_in_use_2.expanding().std()
factor_value_2 = (factor_in_use_2-factors_mean_2) / factors_std_2
print("----------Method 2 ------尝试 cumsum() 和 expanding()-----{}".format(time.time()-start_2))
factor_value_2.to_csv('factor_value_2.csv')

# 尝试3 -  指定的窗口rolling, 这个数值是固定窗口内的zscore的值，而不是累积的所有先前的值.
factor_value_3 = pd.DataFrame()
factor_in_use_3 = factor_in_use.copy()
start_3 = time.time()
window = 80
factors_mean_3 = factor_in_use_3.rolling(window=window, min_periods=1).sum() / np.arange(1, factor_in_use_3.shape[0] + 1)[:, np.newaxis]
factors_std_3 = factor_in_use_3.rolling(window=window, min_periods=1).std()
factor_value_3 = (factor_in_use_3 - factors_mean_3) / factors_std_3
print("----------Method 3 ------指定的窗口rolling-----{}".format(time.time()-start_3))
factor_value_3.to_csv('factor_value_3.csv')


factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
factor_value = factor_value.reset_index()
factor_value.rename(columns={'index':'timestamp'}, inplace=True)
print(factor_value)
factor_value.to_csv('factor_value_file_510050.csv')

# 第四部分：使用matplotlib画图显示
data_15mins_plot = data_15mins[-1000:]
data_15mins_plot = data_15mins_plot.reset_index()
highs = data_15mins_plot['high']
lows = data_15mins_plot['low']
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111, ylabel='stock price')
highs.plot(ax=ax1, color='c', lw=2.)
lows.plot(ax=ax1, color='y', lw=2.)
plt.hlines(highs.head(200).max(),lows.index.values[0],lows.index.values[-1], linewidth=1, color='g')
plt.hlines(lows.head(200).min(),lows.index.values[0],lows.index.values[-1], linewidth=1, color='r')
plt.axvline(linewidth=2,color='b',x=lows.index.values[200],linestyle=':')
plt.legend()
plt.grid()
plt.show()
