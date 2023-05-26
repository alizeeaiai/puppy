import pandas as pd
from pytdx.hq import TdxHq_API
import time


# 计时装饰器
def timer(k_name):
    def decorator(func):
        def wrapper(*args):
            start = time.time()
            func(*args)
            end = time.time()
            print('get ' + k_name + ' data time cost:---------', end - start, 's----------')
        return wrapper
    return decorator


@timer('50etf')
def get_rawdata_day_50etf(local_path, origin_filename, new_filename):

    origin_data = pd.read_excel(local_path + origin_filename).reset_index(drop=True)  # 读excel和csv都需要加上drop=True，因为会把原index保留为一列；而读pickle不需要
    origin_data.rename(columns={'etime': 'timestamp'}, inplace=True)
    origin_data = origin_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    origin_data['timestamp'] = pd.to_datetime(origin_data['timestamp']).dt.date

    api = TdxHq_API()
    if api.connect('119.147.212.81', 7709):
        current_data = api.to_df(api.get_security_bars(4, 1, '510050', 0, 300))
        api.disconnect()

    current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
    current_data.rename(columns={'datetime': 'timestamp', 'vol': 'volume'}, inplace=True)
    current_data['timestamp'] = pd.to_datetime(current_data['timestamp']).dt.date

    data_50etf = pd.concat([origin_data, current_data], axis=0)
    data_50etf = data_50etf.sort_values(by='timestamp', ascending=True)
    data_50etf = data_50etf.drop_duplicates('timestamp').reset_index(drop=True)
    data_50etf = data_50etf.set_index('timestamp')

    # data_50etf.to_excel(local_path + new_filename + '.xlsx')
    # data_50etf.to_csv(local_path + new_filename + '.csv')
    # data_50etf.to_pickle(local_path + new_filename + '.pkl')


if __name__ == '__main__':
    raw_data_path = '/Users/aiailyu/PycharmProjects/第2节课/510050.SH_15.csv'
    origin_file = '510050_d.xlsx'
    new_file = '510050_d_new'
    get_rawdata_day_50etf(raw_data_path, origin_file, new_file)


