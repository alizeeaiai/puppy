import pytdx
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
import pandas as pd
import numpy as np

rawdata = pd.read_excel('/Users/aiailyu/PycharmProjects/第2节课/1_510050_d.xlsx')
rawdata.rename(columns={'etime':'timestamp'}, inplace=True)  # 对其中的列进行重新命名
rawdata['timestamp'] = pd.to_datetime(rawdata['timestamp']).dt.date

api = TdxHq_API()
if api.connect('119.147.212.81', 7709): # 注意这里的IP地址和数据接口
    current_data = api.to_df(api.get_security_bars(4, 1, '510050', 0, 500)) # 注意这里，第一个1表示是15分钟的数据，其中0为5分钟K线 1 15分钟K线 2 30分钟K线 3 1小时K线 4 日K线
    api.disconnect() # 调用完以后一定要关闭接口

current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'amount', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'amount', 'volume']
current_data['timestamp'] = pd.to_datetime(current_data['timestamp']).dt.date

data = pd.concat([rawdata, current_data], axis=0) # 合并数据
data = data.sort_values(by='timestamp', ascending=True) # 注意这一步是非常必要的，要以timestamp作为排序基准
data = data.drop_duplicates('timestamp').reset_index()


del data['index']
data = data.set_index('timestamp')
data = data.replace([np.nan, np.inf, -np.inf], 0.0)
print(data)

# data.to_excel('510050daily.xlsx') # 将文件保存为Excel格式文件
# data.to_csv('510050daily.csv')
# data.to_pickle('510050daily.pkl') # 将文件保存为Excel格式文件
