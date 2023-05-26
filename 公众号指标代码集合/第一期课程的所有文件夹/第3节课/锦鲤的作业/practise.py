import pandas as pd
import numpy as np
from Transaction import Transaction
from Analyse import Analyse
import talib as ta
import warnings
warnings.filterwarnings('ignore')


# 均线趋势跟踪策略
# result_path : 测试结果保存目录
# test_code : 标的代码
# short_length : 短周期均线周期
# median_length : 中周期均线周期
# long_length : 长周期均线周期
# loss_ratio : 止损幅度
# natr : 吊灯止损N倍ATR
def MA_Strategy(result_path, df_price, test_code,
                short_length=5, median_length=10, long_length=20,
                loss_ratio=0.05, natr=3):
    df_price = df_price.copy()
    ts = Transaction(df_price)

    # 计算均线指标
    ts.df_price['sma'] = ta.MA(ts.df_price['close'], timeperiod=short_length, matype=0)
    ts.df_price['mma'] = ta.MA(ts.df_price['close'], timeperiod=median_length, matype=0)
    ts.df_price['lma'] = ta.MA(ts.df_price['close'], timeperiod=long_length, matype=0)

    # 计算ATR通道指标
    ts.df_price['atr'] = ta.ATR(
        np.array(ts.df_price['high']),
        np.array(ts.df_price['low']),
        np.array(ts.df_price['close']),
        timeperiod=20
    )

    # 计算长周期均线5周期斜率
    ts.df_price['lma_slope'] = ta.LINEARREG_SLOPE(ts.df_price['lma'], timeperiod=5)

    # 计算KDJ指标
    ts.df_price['kdj_k'], ts.df_price['kdj_d'] = ta.STOCH(
        ts.df_price['high'],
        ts.df_price['low'],
        ts.df_price['close'],
        fastk_period=9,
        slowk_period=5,
        slowk_matype=1,
        slowd_period=5,
        slowd_matype=1
    )
    ts.df_price['kdj_k_ma5'] = ta.MA(ts.df_price['kdj_k'], timeperiod=5, matype=0)

    # 计算CDP指标
    '''
    CH:=REF(H,1);
    CL:=REF(L,1);
    CC:=REF(C,1);
    CDP:(CH+CL+CC)/3;
    AH:2*CDP+CH-2*CL;
    NH:CDP+CDP-CL;
    NL:CDP+CDP-CH;
    AL:2*CDP-2*CH+CL;
    '''
    ts.df_price['CH'] = ts.df_price['high'].shift(1)
    ts.df_price['CL'] = ts.df_price['low'].shift(1)
    ts.df_price['CC'] = ts.df_price['close'].shift(1)
    ts.df_price['CDP'] = (ts.df_price['CH'] + ts.df_price['CL'] + ts.df_price['CC']) / 3
    ts.df_price['AH'] = 2 * ts.df_price['CDP'] + ts.df_price['CH'] - 2 * ts.df_price['CL']
    ts.df_price['AL'] = 2 * ts.df_price['CDP'] - 2 * ts.df_price['CH'] + ts.df_price['CL']

    # 止损标志
    stoploss_flag = 0

    # 止损价格
    stoploss_price = 0

    # 计步器
    entry_index = 0

    # 入场价格
    entry_price = 0

    # 开始回测
    print('<==========开始测试: %s 回测时间为: %s - %s ============>' % (test_code, df_price.index[0], df_price.index[-1]))
    for i in range(max(1, long_length + 7), ts.df_price.shape[0]):

        # 每天更新一下当天的持仓情况
        if stoploss_flag != 1:
            ts.keepPosition(i)
        else:
            stoploss_flag = 0

        # 短周期均线金叉长周期均线，做多
        long_cond_1 = (ts.queryPosition(i) <= 0) and \
                      (ts.df_price['sma'][i - 1] <= ts.df_price['lma'][i - 1]) and \
                      (ts.df_price['sma'][i] > ts.df_price['lma'][i])

        # 当前无持仓，长周期斜率向上，短周期均线死叉中周期均线，并且短周期均线在长周期均线之上，做多
        long_cond_2 = (ts.queryPosition(i) == 0) and \
                      (ts.df_price['lma_slope'][i] > 0) and \
                      (ts.df_price['sma'][i - 1] >= ts.df_price['mma'][i - 1]) and \
                      (ts.df_price['sma'][i] < ts.df_price['mma'][i]) and \
                      (ts.df_price['sma'][i] > ts.df_price['lma'][i])

        # 当前无持仓，KDJ_K5日均线大于80，昨日KDJ_K大于80，今日KDJ_K小于80，回调做多
        long_cond_3 = (ts.queryPosition(i) == 0) and \
                      (ts.df_price['kdj_k_ma5'][i] > 80) and \
                      (ts.df_price['kdj_k'][i - 1] > 80) and \
                      (ts.df_price['kdj_k'][i] < 80)

        # 短周期均线死叉长周期均线，做空
        short_cond_1 = (ts.queryPosition(i) >= 0) and \
                       (ts.df_price['sma'][i - 1] >= ts.df_price['lma'][i - 1]) and \
                       (ts.df_price['sma'][i] < ts.df_price['lma'][i])

        # 当前无持仓，长周期斜率向下，短周期均线金叉中周期均线，并且短周期均线在长周期均线之下，做空
        short_cond_2 = (ts.queryPosition(i) == 0) and \
                       (ts.df_price['lma_slope'][i] < 0) and \
                       (ts.df_price['sma'][i - 1] < ts.df_price['mma'][i - 1]) and \
                       (ts.df_price['sma'][i] > ts.df_price['mma'][i]) and \
                       (ts.df_price['sma'][i] < ts.df_price['lma'][i])

        # 当前无持仓，KDJ_K5日均线小于20，昨日KDJ_K小于20，今日KDJ_K大于20，回调做空
        short_cond_3 = (ts.queryPosition(i) == 0) and \
                       (ts.df_price['kdj_k_ma5'][i] < 20) and \
                       (ts.df_price['kdj_k'][i - 1] < 20) and \
                       (ts.df_price['kdj_k'][i] > 20)

        # 当前有多头持仓，收盘价跌破止损价格止损
        long_stoploss_cond_1 = (ts.queryPosition(i) > 0) and (ts.df_price['close'][i] < stoploss_price)

        # 当前有空头持仓，收盘价涨破止损价格止损
        short_stoploss_cond_1 = (ts.queryPosition(i) < 0) and (ts.df_price['close'][i] > stoploss_price)

        # 当前有多头持仓，收盘价从最高点回撤natr止损
        long_stoploss_cond_2 = (ts.queryPosition(i) > 0) and \
                               (max(ts.df_price.high[i-5:i]) - ts.df_price['close'][i] > natr * ts.df_price['atr'][i-1]) and \
                               (i - entry_index > 5)

        # 当前有空头持仓，收盘价从最低点回撤natr止损
        short_stoploss_cond_2 = (ts.queryPosition(i) < 0) and \
                                (ts.df_price['close'][i] - min(ts.df_price.low[i - 5:i]) > natr * ts.df_price['atr'][i-1]) and \
                                (i - entry_index > 5)

        # 当前有多头持仓，进场3个bar之后仍旧无法盈利，止损平仓
        long_stoploss_cond_3 = (ts.queryPosition(i) > 0) and (i - entry_index > 3) and \
                               (ts.df_price['close'][i] / entry_price < 0.985)

        # 当前有空头持仓，进场3个bar之后仍旧无法盈利，止损平仓
        short_stoploss_cond_3 = (ts.queryPosition(i) < 0) and (i - entry_index > 3) and \
                                (entry_price / df_price['close'][i] < 0.985)

        # 当前有多头持仓，收盘价跌破CDP指标的AL止损
        long_stoploss_cond_4 = (ts.queryPosition(i) > 0) and (ts.df_price['close'][i] < ts.df_price['AL'][i])

        # 当前有空头持仓，收盘价突破CDP指标的AH止损
        short_stoploss_cond_4 = (ts.queryPosition(i) < 0) and (ts.df_price['close'][i] > ts.df_price['AH'][i])

        if long_cond_1:
            # 有空头持仓先买入平仓
            if ts.queryPosition(i) < 0:
                ts.excuteTrans(i, 'buy', 'close', 0, ts.df_price['close'][i], '短周期均线金叉长周期均线，空头平仓')
            ts.excuteTrans(i, 'buy', 'open', 1, ts.df_price['close'][i], '短周期均线金叉长周期均线，做多')
            stoploss_price = ts.df_price['close'][i] * (1 - loss_ratio)
            entry_index = i
            entry_price = ts.df_price['close'][i]

        elif long_cond_2:
            ts.excuteTrans(i, 'buy', 'open', 1, ts.df_price['close'][i],
                           '当前无持仓，长周期斜率向上，短周期均线死叉中周期均线，并且短周期均线在长周期均线之上，做多')
            stoploss_price = ts.df_price['close'][i] * (1 - loss_ratio)
            entry_index = i
            entry_price = ts.df_price['close'][i]

        elif long_cond_3:
            ts.excuteTrans(i, 'buy', 'open', 1, ts.df_price['close'][i],
                           '当前无持仓，KDJ_K5日均线大于80，昨日KDJ_K大于80，今日KDJ_K小于80，回调做多')
            stoploss_price = ts.df_price['close'][i] * (1 - loss_ratio)
            entry_index = i
            entry_price = ts.df_price['close'][i]

        elif short_cond_1:
            # 有多头持仓先卖出平仓
            if ts.queryPosition(i) > 0:
                ts.excuteTrans(i, 'sell', 'close', 0, ts.df_price['close'][i], '短周期均线死叉长周期均线，多头平仓')
            ts.excuteTrans(i, 'sell', 'open', -1, ts.df_price['close'][i], '短周期均线死叉长周期均线，做空')
            stoploss_price = ts.df_price['close'][i] * (1 + loss_ratio)
            entry_index = i
            entry_price = ts.df_price['close'][i]

        elif short_cond_2:
            ts.excuteTrans(i, 'sell', 'open', -1, ts.df_price['close'][i],
                           '当前无持仓，长周期斜率向下，短周期均线金叉中周期均线，并且短周期均线在长周期均线之下，做空')
            stoploss_price = ts.df_price['close'][i] * (1 + loss_ratio)
            entry_index = i
            entry_price = ts.df_price['close'][i]

        elif short_cond_3:
            ts.excuteTrans(i, 'sell', 'open', -1, ts.df_price['close'][i],
                           '当前无持仓，KDJ_K5日均线小于20，昨日KDJ_K小于20，今日KDJ_K大于20，回调做空')
            stoploss_price = ts.df_price['close'][i] * (1 + loss_ratio)
            entry_index = i
            entry_price = ts.df_price['close'][i]

        elif long_stoploss_cond_1:
            ts.excuteTrans(i, 'sell', 'close', 0, ts.df_price['close'][i], '当前有多头持仓，收盘价跌破止损价格止损')
            stoploss_flag = 1
            stoploss_price = 0

        elif short_stoploss_cond_1:
            ts.excuteTrans(i, 'buy', 'close', 0, ts.df_price['close'][i], '当前有空头持仓，收盘价涨破止损价格止损')
            stoploss_flag = 1
            stoploss_price = 0

        elif long_stoploss_cond_2:
            ts.excuteTrans(i, 'sell', 'close', 0, ts.df_price['close'][i], '当前有多头持仓，收盘价从最高点回撤natr止损')
            stoploss_flag = 1
            stoploss_price = 0

        elif short_stoploss_cond_2:
            ts.excuteTrans(i, 'buy', 'close', 0, ts.df_price['close'][i], '当前有空头持仓，收盘价从最低点回撤natr止损')
            stoploss_flag = 1
            stoploss_price = 0

        elif long_stoploss_cond_3:
            ts.excuteTrans(i, 'sell', 'close', 0, ts.df_price['close'][i], '当前有多头持仓，进场3个bar之后仍旧无法盈利，止损平仓')
            stoploss_flag = 1
            stoploss_price = 0

        elif short_stoploss_cond_3:
            ts.excuteTrans(i, 'buy', 'close', 0, ts.df_price['close'][i], '当前有空头持仓，进场3个bar之后仍旧无法盈利，止损平仓')
            stoploss_flag = 1
            stoploss_price = 0

        elif long_stoploss_cond_4:
            ts.excuteTrans(i, 'sell', 'close', 0, ts.df_price['close'][i], '当前有多头持仓，收盘价跌破CDP指标的AL止损')
            stoploss_flag = 1
            stoploss_price = 0

        elif short_stoploss_cond_4:
            ts.excuteTrans(i, 'buy', 'close', 0, ts.df_price['close'][i], '当前有空头持仓，收盘价突破CDP指标的AH止损')
            stoploss_flag = 1
            stoploss_price = 0

    # 保存回测数据
    df_price, order_list, merge_order = ts.statistics(long_length)
    df_price.to_csv(result_path + test_code + '回测过程数据.csv')
    order_list.to_csv(result_path + test_code + '交易订单数据.csv')
    merge_order.to_csv(result_path + test_code + '交易订单合并数据.csv')

    # 将过程数据和订单数据合并，用于观察交易是否正常
    all_data = pd.merge(df_price, order_list, left_index=True, right_on='日期', how='left')
    all_data.set_index('日期', inplace=True)
    all_data.to_csv(result_path + test_code + ' All Test data.csv')

    # 策略分析
    analyse = Analyse(df_price, order_list, merge_order)
    analyse.show_performance()
    analyse.plot(result_path, test_code + '净值曲线图.html')


# 策略说明
# 测试条件：
#   1、交易成本为双边万5
#   2、信号出现收盘价入场
#   3、无风险收益率为4%
# 资金管理：
#   1、合约市值始终保持账户资金的100%（即不带杠杆）
if __name__ == '__main__':
    data_path = 'E:/Quant/lesson2/practise/data/'
    result_path = 'E:/Quant/lesson2/practise/practise result/'
    file_list = ['CU.csv', 'I.csv', 'L.csv', 'M.csv', 'RB.csv',
                 'RU.csv', 'SR.csv', 'TA.csv', 'Y.csv', 'ZN.csv']
    # file_list = ['CU.csv']
    for file_name in file_list:
        df_price = pd.read_csv(data_path + file_name)
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        df_price = df_price.set_index('timestamp')
        test_code = file_name.split('.')[0]
        MA_Strategy(result_path, df_price, test_code)
