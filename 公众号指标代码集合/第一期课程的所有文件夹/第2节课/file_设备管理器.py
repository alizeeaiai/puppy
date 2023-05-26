# -*- codeing = utf-8 -*-
# @Time : 2023/2/27 14:13
# @Author : EquipmentADV
# @File : Lesson_1_test.py
# @Software : PyCharm
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import pytdx
from pytdx.hq import TdxHq_API


def preRead(filePath):
    pre_df = pd.read_excel(filePath, sheet_name = "file")
    pre_df.rename(columns = {'etime': 'timestamp'}, inplace = True)
    pre_df['timestamp'] = pd.to_datetime(pre_df['timestamp']) + pd.Timedelta(15, "H")
    # print("\n", type(pre_df['timestamp'][0]))
    # print(pre_df['timestamp'][0])
    # print(pre_df)
    return  pre_df

# data_path = "/Users/aiailyu/PycharmProjects/第2节课/"  # 将源文件夹地址命名为data_path
# try:
#     data_1d = pd.read_excel(data_path + '1_510050_d.xlsx')  # 读取csv文件
# except FileNotFoundError:
#     print("Not found file: ", data_path + '510050.SH_1d.csv', " , File is created now.")
#     new_data = pd.DataFrame(columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     new_data.set_index('timestamp', inplace = True)
#     # print(new_data)
#     new_data.to_csv(data_path + '510050.SH_1d.csv')
#     data_1d = pd.read_csv(data_path + '510050.SH_1d.csv')  # 读取csv文件

# data_1d = data_1d[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
# data_1d['timestamp'] = pd.to_datetime(data_1d['timestamp'])
data_1d = preRead(r"/Users/aiailyu/PycharmProjects/第2节课/1_510050_d.xlsx")

apiList = [("招商证券深圳行情", "119.147.212.81", 7709),
           ("华泰证券(南京电信)", "221.231.141.60", 7709),
           ("华泰证券(上海电信)", "101.227.73.20", 7709),
           ("华泰证券(上海电信二)", "101.227.77.254", 7709),
           ("华泰证券(深圳电信)", "14.215.128.18", 7709),
           ("华泰证券(武汉电信)", "59.173.18.140", 7709),
           ("华泰证券(天津联通)", "60.28.23.80", 7709),
           ("华泰证券(沈阳联通)", "218.60.29.136", 7709),
           ("华泰证券(南京联通)", "122.192.35.44", 7709),
           ("华泰证券(南京联通)", "122.192.35.44", 7709)]

api = TdxHq_API()
for aipConnect in apiList:
    if api.connect(aipConnect[1], aipConnect[2]):  # 注意这里的IP地址和数据接口
        print(aipConnect[0], " Connect succeeded")
        current_data = api.to_df(api.get_security_bars(4, 1, '510050', 0, 300))
        api.disconnect()
        break

current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])

data_1d = pd.concat([data_1d, current_data], axis = 0)
data_1d = data_1d.sort_values(by = 'timestamp', ascending = True)
data_1d = data_1d.drop_duplicates('timestamp').reset_index()
del data_1d['index']
data_1d.set_index('timestamp', inplace = True)
print(data_1d)
# data_1d.to_csv(data_path + '510050.SH_1d.csv')
# data_1d.to_excel(data_path + '510050.SH_1d.xlsx')
# data_1d.to_pickle(data_path + '510050.SH_1d.pkl')
