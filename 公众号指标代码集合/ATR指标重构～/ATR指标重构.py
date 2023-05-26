import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API

import talib as ta
import time
import datetime
import warnings

warnings.filterwarnings('ignore')


def data_process():
    file_path = './宁德时代每日成交金额.csv'
    df = pd.read_csv(file_path)

    df = df.drop(['Unnamed: 0', 'ts_code', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1)
    df['timestamp'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values(by='timestamp', ascending=True)
    df.reset_index(drop=True, inplace=True)
    df['support'] = 0
    df['trends'] = 0
    df = df.drop(['trade_date'], axis=1)
    return df


# data是BTC的OLHCVM数据，也就是包含开盘价、收盘价、最高和最低价、成交量的数据
def support(df):

    for i in range(4, df.shape[0]):
        if df['low'][i] >= df['low'][i-2] and df['low'][i-1] >= df['low'][i-2] and df['low'][i-3] >= df['low'][i-2] and df['low'][i-4] >= df['low'][i-2]:
            df['support'][i] = df.low[i-2]

        elif df['low'][i] > df['high'][i-1] * 1.0013:
            df['support'][i] = df['high'][i - 1] * 0.9945

        elif df['low'][i] > df['support'][i-1] * 1.1:
            df['support'][i] = df['support'][i - 1] * 1.05

        else:
            df['support'][i] = df['support'][i-1]
    return df


def resistance(df):
    df['H-L'] = df['high'] - df['low']
    df['SMA'] = df['H-L'].rolling(window=10).mean()
    df['HiLo'] = np.where(df['H-L'] < 1.5 * df['SMA'], df['H-L'], 1.5 * df['SMA'])

    # 计算Href和Lref值
    df['Href'] = np.where(df['low'] <= df['high'].shift(-1), df['high'] - df['close'].shift(-1),
                          (df['high'] - df['close'].shift(-1)) - (df['low'].shift(-1) - df['high'].shift(-1)) / 2)
    df['Lref'] = np.where(df['high'] >= df['low'].shift(-1), df['close'].shift(-1) - df['low'],
                          (df['close'].shift(-1) - df['low']) - (df['low'].shift(-1) - df['high'].shift(-1)) / 2)

    # 计算diff1、diff2和ATRmod值
    df['diff1'] = np.maximum(df['HiLo'], df['Href'])
    df['diff2'] = np.maximum(df['diff1'], df['Lref'])
    df['ATRmod'] = df['diff2'].ewm(alpha=1 / 10, adjust=False).mean()

    # 计算resistance值
    df['loss'] = 2.8 * df['ATRmod']
    df['resistance'] = df['close'] + df['loss']
    return df


def trends(df):
    for i in range(1, df.shape[0]):
        if df['high'][i] > df['support'][i-1] and df['high'][i-1] > df['support'][i-1]:
            df['trends'][i] = max(df['support'][i-1], df['support'][i])
        elif df['high'][i] < df['support'][i-1] and df['high'][i-1] < df['support'][i-1]:
            df['trends'][i] = min(df['support'][i-1], df['resistance'][i])
        elif df['high'][i] >= df['support'][i-1]:
            df['trends'][i] = df['support'][i]
        else:
            df['trends'][i] = df['resistance'][i]

    df.to_csv('./trends_added.csv')


if __name__ == '__main__':
    data = data_process()
    data = support(data)
    data = resistance(data)
    data = trends(data)
