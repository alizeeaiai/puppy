import time

import numpy as np
import pandas as pd
import openpyxl
import pytdx
from pytdx.hq import TdxHq_API

#pycharm控制台输出不全
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

start_time = time.time()
data_path = '/Users/aiailyu/PycharmProjects/第2节课/'
data_day = pd.read_excel(data_path + '510050_d.xlsx')
# del data_day['amount']

data_day.rename(columns={'etime':'timestamp'}, inplace=True)
data_day['timestamp'] = pd.to_datetime(data_day['timestamp'])


api = TdxHq_API()
if api.connect('119.147.212.81', 7709):
    current_data = api.to_df(api.get_security_bars(4, 1, '510050', 0, 300))
    api.disconnect()


current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close',  'volume']
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])

data_day = pd.concat([data_day, current_data], axis=0)
data_day = data_day.sort_values(by='timestamp', ascending=True)
data_day = data_day.drop_duplicates('timestamp').reset_index()

del data_day['index']
data_day = data_day.set_index('timestamp')
print(data_day)

data_day.to_csv(data_path + '510050_d.csv')
data_day.to_excel(data_path + '510050_d.xlsx')
data_day.to_pickle(data_path + '510050_d.pkl')




