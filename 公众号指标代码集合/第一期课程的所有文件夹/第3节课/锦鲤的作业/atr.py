import pandas as pd
import numpy as np
from Transaction import Transaction
from Analyse import Analyse
import talib as ta
import warnings
warnings.filterwarnings('ignore')


# 超级趋势线策略
# result_path : 测试结果保存目录
# test_code : 标的代码
# natr : N倍ATR
# timeperiod : ATR周期
def supertrend_strategy(result_path, df_price, test_code, natr=1, timeperiod=20):
    df_price = df_price.copy()
    ts = Transaction(df_price)

    # 计算ATR指标
    ts.df_price['atr'] = ta.ATR(
        np.array(ts.df_price['high']),
        np.array(ts.df_price['low']),
        np.array(ts.df_price['close']),
        timeperiod=timeperiod
    )

    # 计算上下轨
    ts.df_price['upper'] = ((ts.df_price['high'] + ts.df_price['low']) / 2) + natr * ts.df_price['atr']
    ts.df_price['lower'] = ((ts.df_price['high'] + ts.df_price['low']) / 2) - natr * ts.df_price['atr']

    # 开始回测
    print('<==========开始测试: %s 回测时间为: %s - %s ============>' % (test_code, df_price.index[0], df_price.index[-1]))
    for i in range(max(1, timeperiod+2), ts.df_price.shape[0]):

        # 每天更新一下当天的持仓情况
        ts.keepPosition(i)

        # 向上突破上轨做多
        long_cond_1 = (ts.queryPosition(i) <= 0) and (ts.df_price['close'][i] > ts.df_price['upper'][i-1])

        # 向下突破下轨做空
        short_cond_1 = (ts.queryPosition(i) >= 0) and (ts.df_price['close'][i] < ts.df_price['lower'][i-1])

        if long_cond_1:
            # 有空头持仓先买入平仓
            if ts.queryPosition(i) < 0:
                ts.excuteTrans(i, 'buy', 'close', 0, ts.df_price['close'][i], '突破上轨，空头平仓')
            # 突破上轨之后开多仓
            ts.excuteTrans(i, 'buy', 'open', 1, ts.df_price['close'][i], '突破上轨做多')

        elif short_cond_1:
            # 有多头持仓先卖出平仓
            if ts.queryPosition(i) > 0:
                ts.excuteTrans(i, 'sell', 'close', 0, ts.df_price['close'][i], '突破下轨，多头平仓')
            # 突破下轨做空
            ts.excuteTrans(i, 'sell', 'open', -1, ts.df_price['close'][i], '突破下轨做空')

    # 保存回测数据
    df_price,order_list,merge_order = ts.statistics(timeperiod)
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
# 超级趋势线策略
# 指标计算公式：
#   upper = (High + Low) / 2 + NATR
#   lower = (High + Low) / 2 - NATR
# 测试条件：
#   1、交易成本为双边万5
#   2、信号出现收盘价入场
#   3、无风险收益率为4%
# 进出场方式:
#   1、突破上轨，平空单，开多单
#   2、突破下轨，平多单，开空单
# 资金管理：
#   1、合约市值始终保持账户资金的100%（即不带杠杆）
if __name__ == '__main__':
    data_path = 'E:/Quant/lesson2/practise/data/'
    result_path = 'E:/Quant/lesson2/practise/supertrend result/'
    file_list = ['CU.csv', 'I.csv', 'L.csv', 'M.csv', 'RB.csv',
                 'RU.csv', 'SR.csv', 'TA.csv', 'Y.csv', 'ZN.csv']
    # file_list = ['CU.csv']
    for file_name in file_list:
        df_price = pd.read_csv(data_path + file_name)
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        df_price = df_price.set_index('timestamp')
        test_code = file_name.split('.')[0]
        supertrend_strategy(result_path, df_price, test_code)


