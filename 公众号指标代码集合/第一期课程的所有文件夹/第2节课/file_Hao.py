
from datetime import datetime
from pytdx.hq import TdxHq_API
import pandas as pd

df = pd.read_csv('/Users/aiailyu/PycharmProjects/第2节课/510050.SH_15.csv')
df['timestamp'] = pd.to_datetime(df['etime'])

api = TdxHq_API()
if api.connect('119.147.212.81', 7709):
    new_add = api.to_df(api.get_security_bars(4, 1, '510050', 0, 500))

    api.disconnect()
new_add.rename(columns={'vol':'volume'}, inplace=True)
new_add['datetime'] = new_add['datetime'].apply(lambda x: x[:10])
new_add['timestamp'] = pd.to_datetime(new_add['datetime'])
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
new_add = new_add[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

df = pd.concat([df, new_add], axis=0)

df = df.sort_values(by='timestamp', ascending=True)
df = df.drop_duplicates('timestamp').set_index('timestamp')

# df.to_csv('510050_update.csv')
# df.to_excel('510050_update.xlsx')
# df.to_pickle('510050_update.pkl')


