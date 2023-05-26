import os
import talib as ta
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
data_base_path = "./lesson2data/"


def MA_Strategy(data, window_short=20, window_long=60):
    # 简单双均线策略
    data_price = data.copy()
    data_price['sma'] = ta.MA(data_price['close'], timeperiod=window_short, matype=0)
    data_price['lma'] = ta.MA(data_price['close'], timeperiod=window_long, matype=0)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['sma'][i - 1] < data_price['lma'][i - 1]) and (
                data_price['sma'][i] > data_price['lma'][i]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            entry_index = i
            # print(data_price.index[i], '=========金叉买入@--', price_in)
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['sma'][i - 1] > data_price['lma'][i - 1]) & (
                data_price['sma'][i] < data_price['lma'][i]):
            # print(data_price.index[i], '=========死叉卖出')
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def MACD_Strategy(data, window_short=9, window_median=12, window_long=26):
    # MACD策略
    data_price = data.copy()
    data_price['macd'], data_price['signal'], data_price['hist'] = ta.MACD(np.array(data_price['close']))
    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['macd'][i - 1] < data_price['signal'][i - 1]) and (
                data_price['macd'][i] > data_price['signal'][i]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['macd'][i - 1] > data_price['signal'][i - 1]) & (
                data_price['macd'][i] < data_price['signal'][i]):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def KDJ_Strategy(data, window_short=3, window_long=9):
    # KDJ策略
    data_price = data.copy()
    data_price['k'], data_price['d'] = ta.STOCH(data_price['high'], data_price['low'], data_price['close'],
                                                fastk_period=9, slowk_period=3, slowd_period=3)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['k'][i - 1] < data_price['d'][i - 1]) and (
                data_price['k'][i] > data_price['d'][i]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['k'][i - 1] > data_price['d'][i - 1]) & (
                data_price['k'][i] < data_price['d'][i]):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def RSI_Strategy(data, window_long=14):
    # RSI策略
    data_price = data.copy()
    data_price['rsi'] = ta.RSI(data_price['close'], timeperiod=window_long)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['rsi'][i - 1] < 30) and (
                data_price['rsi'][i] >= 30):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['rsi'][i - 1] > 70) & (
                data_price['rsi'][i] <= 70):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def DMI_Strategy(data, window_long=14):
    # DMI策略
    data_price = data.copy()
    data_price['adx'] = ta.ADX(data_price['high'], data_price['low'], data_price['close'], timeperiod=window_long)
    data_price['plus_di'] = ta.PLUS_DI(data_price['high'], data_price['low'], data_price['close'],
                                       timeperiod=window_long)
    data_price['minus_di'] = ta.MINUS_DI(data_price['high'], data_price['low'], data_price['close'],
                                         timeperiod=window_long)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['plus_di'][i - 1] < data_price['minus_di'][i - 1]) and (
                data_price['plus_di'][i] > data_price['minus_di'][i]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['plus_di'][i - 1] > data_price['minus_di'][i - 1]) & (
                data_price['plus_di'][i] < data_price['minus_di'][i]):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def BOLL_Strategy(data, window_long=20):
    # BOLL策略
    data_price = data.copy()
    data_price['upper_band'], data_price['middle_band'], data_price['lower_band'] = ta.BBANDS(data_price['close'],
                                                                                              timeperiod=window_long,
                                                                                              nbdevup=2, nbdevdn=2)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (
                data_price['close'][i - 1] >= data_price['lower_band'][i - 1]) and (
                data_price['close'][i] < data_price['lower_band'][i]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['close'][i - 1] <= data_price['upper_band'][i - 1]) & (
                data_price['close'][i] > data_price['upper_band'][i]):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def SAR_Strategy(data):
    # SAR策略
    data_price = data.copy()
    data_price['sar'] = ta.SAR(data_price['high'], data_price['low'], acceleration=0.02, maximum=0.2)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(1, data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['close'][i - 1] >= data_price['sar'][i - 1]) and (
                data_price['close'][i] < data_price['sar'][i]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['close'][i - 1] <= data_price['sar'][i - 1]) & (
                data_price['close'][i] > data_price['sar'][i]):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    # data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def CCI_Strategy(data, window_long=14):
    # CCI策略
    data_price = data.copy()
    data_price['cci'] = ta.CCI(data_price['high'], data_price['low'], data_price['close'], timeperiod=window_long)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['cci'][i - 1] >= -100) and (
                data_price['cci'][i] < -100):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['cci'][i - 1] <= 100) & (
                data_price['cci'][i] > 100):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def WILLR_Strategy(data, window_long=14):
    # 威廉姆斯R策略
    data_price = data.copy()
    data_price['willr'] = ta.WILLR(data_price['high'], data_price['low'], data_price['close'], timeperiod=window_long)

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(max(1, window_long), data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['willr'][i - 1] >= -80) and (
                data_price['willr'][i] < -80):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['willr'][i - 1] <= -20) & (
                data_price['willr'][i] > -20):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def OBV_Strategy(data):
    # OBV策略
    data_price = data.copy()
    data_price['obv'] = ta.OBV(data_price['high'], data_price['volume'])

    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0  # 记录仓位
    data_price['flag'] = 0  # 记录买卖

    for i in range(1, data.shape[0]):

        if (data_price['position'][i - 1] == 0) and (data_price['obv'][i] < data_price['obv'][i - 1]) and (
                data_price['close'][i] > data_price['obv'][i - 1]):
            data_price['flag'][i] = 1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            Buy.append([date_in, price_in, '金叉买入'])

        elif (data_price['position'][i - 1] == 1) & (data_price['obv'][i] > data_price['obv'][i - 1]) & (
                data_price['close'][i] < data_price['obv'][i - 1]):
            data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里
        else:
            data_price['position'][i] = data_price['position'][i - 1]

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # p1和p2合并，axis=1表示以水平方向合并

    # data_price = data_price.iloc[window_long:, :]
    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    return calc_sharp(transactions, data_price)


def calc_sharp(transactions, strategy):
    # 夏普比
    N = 250
    Sharp = (strategy.ret * strategy.position).mean() / (strategy.ret * strategy.position).std() * np.sqrt(N)
    return round(Sharp, 2)


def handle_data(data: pd.DataFrame):
    my_set = []
    data = data.copy()
    data = data.rename(
        columns={"名称": "name", "日期": "timestamp", "开盘价(元)": "open", "最高价(元)": "high", "最低价(元)": "low",
                 "收盘价(元)": "close", "成交量(股)": "volume"})
    name = data['name'].iloc[0]
    data = data.loc[:, ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    data['timestamp'] = data['timestamp'].dt.date
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp', ascending=True)
    data = data.drop_duplicates('timestamp').reset_index()
    del data['index']
    data = data.set_index('timestamp')

    data = data.replace([np.inf, -np.inf, np.nan], 0.0)

    sharp = MA_Strategy(data)
    my_set.append([name, "MA双均线", sharp])
    # print(sharp)
    sharp = MACD_Strategy(data)
    my_set.append([name, "MACD", sharp])
    # print(sharp)
    sharp = KDJ_Strategy(data)
    my_set.append([name, "KDJ", sharp])
    # print(sharp)
    sharp = RSI_Strategy(data)
    my_set.append([name, "RSI", sharp])
    # print(sharp)
    sharp = DMI_Strategy(data)
    my_set.append([name, "DMI", sharp])
    # print(sharp)
    sharp = BOLL_Strategy(data)
    my_set.append([name, "BOLL", sharp])
    # print(sharp)
    sharp = SAR_Strategy(data)
    my_set.append([name, "SAR", sharp])
    # print(sharp)
    sharp = CCI_Strategy(data)
    my_set.append([name, "CCI", sharp])
    # print(sharp)
    sharp = WILLR_Strategy(data)
    my_set.append([name, "WILLR", sharp])
    # print(sharp)
    sharp = OBV_Strategy(data)
    my_set.append([name, "OBV", sharp])
    # print(sharp)
    return my_set


def main():
    # 1、10个品种，10个技术指标，金叉死叉作为买卖条件，用来选择品种；输出最终结果，看看哪几个品种的sharpe最高
    files = os.listdir(data_base_path)
    result = []
    for i in files:
        file_path = data_base_path + i
        data = pd.read_excel(file_path, engine="openpyxl")
        result += handle_data(data)
    result = pd.DataFrame(result, columns=["品种", "策略", "夏普"])
    idx = result.groupby("品种")["夏普"].idxmax()
    result = result.loc[idx][["品种", "策略", "夏普"]].reset_index(drop=True)
    print(result)

    # 2、用slope来表示5个周期的20日均线向上——（1）talib查询slope函数，dataframe里面，生成slope-series
    files = os.listdir(data_base_path)
    file_path = data_base_path + files[0]
    df = pd.read_excel(file_path, engine="openpyxl")
    # 计算20日移动平均线
    close = df['收盘价(元)']
    ma20 = ta.SMA(close, timeperiod=20)

    # 计算20日移动平均线的斜率
    slope = ta.LINEARREG_SLOPE(ma20, timeperiod=5)

    # 判断移动平均线是否向上
    for i in range(5, slope.shape[0]):
        if slope[i] > slope[i - 1]:
            print('20日移动平均线向上')
        else:
            print('20日移动平均线向下')

    # 3、开发N倍ATR作为高点回落出局条件；
    #     算出 data_price['atr']
    #     当收盘价低于5日最高-2倍ATR 出局
    #
    #     elif (data_price['position'][i - 1] == 1) and (
    #         ((max(data_price.high[i - 5:i]) - 2 * data_price['atr'][i] ) < data_price['close'][i]):
    #
    #         data_price['flag'][i] = -1  # 记录买入还是卖出，1是买入
    #         data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
    #         date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
    #         price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
    #         Sell.append([date_out, price_out, '回落平仓'])  # 把卖出记录保存到Sell列表里


if __name__ == '__main__':
    main()
