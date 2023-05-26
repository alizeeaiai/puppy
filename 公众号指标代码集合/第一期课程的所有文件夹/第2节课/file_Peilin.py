import pandas as pd
import pytdx
from pytdx.hq import TdxHq_API
import os

pd.set_option('display.max_rows', None)


def renew_data_1day_50etf(aim_folder):

    api = TdxHq_API()

    data = pd.read_excel(aim_folder)
    data.rename(columns={'etime': 'timestamp'}, inplace=True)
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.date

    if api.connect('119.147.212.81', 7709):

        data_new = api.to_df(
            api.get_security_bars(4, 1, '510050', 0, 180))
        api.disconnect()

    data_new.rename(columns={'vol': 'volume'}, inplace=True)
    data_new.rename(columns={'datetime': 'timestamp'}, inplace=True)

    data_new = data_new[['timestamp', 'open',
                         'high', 'low', 'close', 'volume']]
    data_new['volume'] = data_new['volume'].fillna(0).astype(int)
    data_new['timestamp'] = pd.to_datetime(data_new['timestamp']).dt.date


    data = pd.concat([data, data_new], axis=0)
    data = data.sort_values(by='timestamp', ascending=True)
    data = data.drop_duplicates('timestamp').reset_index()
    data = data.set_index('timestamp')
    data = data[['open', 'high', 'low', 'close', 'volume']]

    data.to_csv(aim_folder + '510050_d_new.csv')
    data.to_excel(aim_folder + '510050_d_new.xlsx')
    data.to_pickle(aim_folder + '510050_d_new.pkl')


if __name__ == '__main__':
    _aim_folder = os.path.join(
        os.getcwd(), '510050_d.xlsx')
    renew_data_1day_50etf(_aim_folder)
