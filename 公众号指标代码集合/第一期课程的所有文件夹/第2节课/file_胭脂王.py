import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from pytdx.hq import TdxHq_API

data_path = '/Users/aiailyu/PycharmProjects/第2节课/'

#读取510050的csv文件
data_510050 = pd.read_excel(data_path + "1_510050_d.xlsx")

#替换列名，清洗时间数据
data_510050.rename(columns = {'etime':'timestamp'},inplace=True)
data_510050['timestamp'] = pd.to_datetime(data_510050['timestamp'] + datetime.timedelta(hours=15)) #原始数据是0：00，通达信数据是15：00，因此原始数据需要加上15小时
data_510050 = data_510050[['timestamp','open','high','low','close','volume']]

#通达信读取数据 https://gitee.com/better319/pytdx

api = TdxHq_API()
if api.connect('119.147.212.81', 7709):
    current_data = api.get_security_bars(4, 1, '510050', 0, 500)
    current_data = api.to_df(current_data)
    api.disconnect()

current_data = current_data[['datetime','open','high','low','close','vol']] #提取部分数据
current_data.columns = ['timestamp','open','high','low','close','volume'] #修改列名称
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])




#文件二合一
data_510050 = pd.concat([data_510050,current_data],axis=0)
data_510050 = data_510050.sort_values(by='timestamp',ascending=True)
data_510050 = data_510050.drop_duplicates('timestamp').reset_index()
data_510050 = data_510050.set_index('timestamp')
del data_510050['index']
print(data_510050)

# #将文件保存为excel、csv、plk格式
# data_510050.to_csv(data_path + "510050test.csv")
# data_510050.to_excel(data_path + "510050test2.xlsx")
# data_510050.to_pickle(data_path + "510050test3.pkl")


