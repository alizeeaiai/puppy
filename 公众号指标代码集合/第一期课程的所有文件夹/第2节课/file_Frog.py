import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import warnings
warnings.filterwarnings('ignore')

data_path = '/Users/aiailyu/PycharmProjects/第2节课/'
data = pd.read_csv(data_path + '510050_15mins.csv')

data.rename(columns={'etime': 'timestamp'}, inplace=True)
data['timestamp'] = pd.to_datetime(data['timestamp'])

import pytdx
from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
api = TdxHq_API()



# 15分钟线，每天4小时共16条数据,因此缺失的最新数据不超过 自然日差*16条

k_line_amount = (pd.Timestamp('now') - pd.to_datetime('2022-11-29')).days*16


with api.connect('119.147.212.81', 7709):

    current_data = pd.DataFrame()
    i = 0

    while i < k_line_amount:
        new_data = api.to_df(api.get_security_bars(1, 1, '510050',i, 500))
        current_data = pd.concat([current_data, new_data], axis=0)
        i = i +500


current_data = current_data[['datetime','open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])

complete_data = pd.concat([data, current_data], axis=0)
complete_data = complete_data.sort_values(by='timestamp', ascending=True)
complete_data = complete_data.drop_duplicates('timestamp').reset_index()

del complete_data['index']
complete_data = complete_data.set_index('timestamp')
print(complete_data)
#
# complete_data.to_csv(data_path + 'complete_data.csv')
# complete_data.to_excel(data_path + 'complete_data.xlsx')
# complete_data.to_pickle(data_path + 'complete_data.pkl')
