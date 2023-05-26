import warnings
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from sklearn import metrics as me
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
import time
import talib as ta
warnings.filterwarnings('ignore')


def prepross():
    df = pd.read_csv('./BTCUSDT_spot_2017-2023_1h.csv')
    # 修改df列名，变成标准的数据格式
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    # 新增一列，计算收益率
    df['return'] = df.close.pct_change(1).shift(-1)
    df.dropna(axis=0, how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['ma'] = ta.MA(df.close, 10)
    # 10日均线前10个值是空的要dropna
    df = df.replace([np.nan], 0)


    # 这一步，搞半天，是要把每一列转化为float？？？？？
    fields = df.columns[1:]
    for each_column in fields:
        df[each_column] = df[each_column].values.astype('float')
    print(df)
    # 把x train变成ndarray，这是一个49356行，每行有4列的ndarray
    x_train = df.drop(columns=['timestamp', 'return']).to_numpy()
    y_train = df['return'].values

    # 添加一些其他的数据，作为底层算法


if __name__ == '__main__':
    prepross()