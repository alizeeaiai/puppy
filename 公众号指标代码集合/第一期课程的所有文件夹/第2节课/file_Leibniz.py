import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


start_time = time.time()
data_path = '/Users/aiailyu/PycharmProjects/第2节课/' # 将源文件夹地址命名为data_path
data_15mins = pd.read_csv(data_path + '510050.SH_15.csv') # 读取csv文件


data_15mins.rename(columns={'etime':'timestamp'}, inplace=True)  # 对其中的列进行重新命名
data_15mins['timestamp'] = pd.to_datetime(data_15mins['timestamp'])

import pytdx
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API

api = TdxHq_API()
if api.connect('119.147.212.81', 7709): # 注意这里的IP地址和数据接口
    current_data = api.to_df(api.get_security_bars(4, 1, '510050', 0, 500)) # 注意这里，第一个1表示是15分钟的数据，其中0为5分钟K线 1 15分钟K线 2 30分钟K线 3 1小时K线 4 日K线
    api.disconnect() # 调用完以后一定要关闭接口

current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])

data = pd.concat([data_15mins, current_data], axis=0)
data = data.sort_values(by='timestamp', ascending=True)
data = data.drop_duplicates('timestamp').reset_index()
print(data)

del data['index']
data = data.set_index('timestamp')

# data.to_csv(data_path + '510050_15mins.csv')
# data.to_excel(data_path + '510050.SH_15mins.xlsx')
# data.to_pickle(data_path + '510050.SH_15mins.pkl')




