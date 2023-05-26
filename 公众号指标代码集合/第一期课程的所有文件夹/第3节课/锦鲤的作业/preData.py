import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime


# 数据预处理
def pre_data():
    origin_data_path = 'E:/Quant/lesson2/十种大宗商品指数文件/'
    new_data_path = 'E:/Quant/lesson2/practise/data/'
    file_list = ['CU.csv', 'I.csv', 'L.csv', 'M.csv', 'RB.csv',
                 'RU.csv', 'SR.csv', 'TA.csv', 'Y.csv', 'ZN.csv']

    for data_file in file_list:
        origin_data = pd.read_csv(origin_data_path + data_file)

        # 修改列名
        origin_data.rename(columns={'trade_date': 'timestamp'}, inplace=True)

        # 保留ohlc字段
        origin_data = origin_data.iloc[:, 1:]
        origin_data['timestamp'] = pd.to_datetime(origin_data['timestamp'])
        origin_data.set_index('timestamp', inplace=True)

        origin_data.to_csv(new_data_path + data_file)


if __name__ == '__main__':
    pre_data()
