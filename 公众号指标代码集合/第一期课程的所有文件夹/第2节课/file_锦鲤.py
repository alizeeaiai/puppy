import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
from pytdx.hq import TdxHq_API


def get_tdx_data():

    # 获取原始表excel数据
    data_path = '/Users/aiailyu/PycharmProjects/第2节课/'
    origin_data = pd.read_excel(data_path + '1_510050_d.xlsx')

    # 将原始表数据处理成标准格式
    origin_data.rename(columns={'etime': 'timestamp'}, inplace=True)
    origin_data['timestamp'] = pd.to_datetime(origin_data['timestamp'])

    # 获取原始表最新日K数据日期
    start_date = origin_data.iloc[-1, 0]
    print('start_date : ', start_date)

    # 接入通达信行情数据
    api = TdxHq_API()
    if api.connect('119.147.212.81', 7709):
        total_data = pd.DataFrame()
        step = 50
        cnt = 0
        while True:
            tdx_data = api.to_df(api.get_security_bars(4, 1, '510050', (step * cnt), 50))
            tdx_data = tdx_data[['datetime', 'open', 'high', 'low', 'close', 'amount', 'vol']]
            tdx_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'amount', 'volume']

            # 将接入的TDX数据的时间戳格式修改为与原文件时间格式一致（去除掉时分秒）
            tdx_data['timestamp'] = pd.to_datetime(tdx_data['timestamp'])
            tdx_data['timestamp'] = tdx_data['timestamp'].dt.date
            tdx_data['timestamp'] = pd.to_datetime(tdx_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
            total_data = pd.concat([total_data, tdx_data], axis=0)

            # 检验当前已接入的数据最早一天是否已经在原始数据的范围之内
            min_date = min(total_data['timestamp'])
            print('min date : ', min_date)
            if min_date < start_date:
                break
            else:
                cnt = cnt + 1
        api.disconnect()


    new_data = pd.concat([origin_data, total_data], axis=0)
    new_data = new_data.sort_values(by='timestamp', ascending=True)
    print(new_data)
    new_data = new_data.drop_duplicates('timestamp').reset_index()
    del new_data['index']
    new_data.set_index('timestamp', inplace=True)
    print(new_data)

    # # 将数据存储到文件
    # new_data.to_csv(data_path + '510050.SH_d.csv')
    # new_data.to_excel(data_path + '510050.SH_d.xlsx')
    # new_data.to_pickle(data_path + '510050.SH_d.pkl')


if __name__ == '__main__':
    get_tdx_data()
