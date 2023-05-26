import numpy as np
import pandas as pd
from pytdx.hq import TdxHq_API
import time
import datetime
import talib as ta
import warnings
from Angel的作业1 import BasicMethods, pre_process, CDP, show_performance

warnings.filterwarnings('ignore')


# 1. 双均线
class DualMovingAverage(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super(DualMovingAverage, self).__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, window_short=5, window_median=10, window_long=20, loss_ratio=0.10, pedometer=3, ratio=0.985):
        '''
        :param window_short: 短均线周期，默认为5；
        :param window_median: 中均线周期，默认为10
        :param window_long: 长均线周期，默认为20；
        :param loss_ratio: 止损率,默认为10%，即开仓后下跌超过10%止损
        :param pedometer: 计步器，买入后超过几个bar不赚钱就平仓
        :param ratio: 计步器不赚钱的标准，默认0.985

        写了几个做空示意
        '''
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['sma'] = ta.MA(data_price['close'], timeperiod=window_short, matype=0)
        data_price['lma'] = ta.MA(data_price['close'], timeperiod=window_long, matype=0)
        data_price['mma'] = ta.MA(data_price['close'], timeperiod=window_median, matype=0)
        data_price['kdj_k'], data_price['kdj_d'] = ta.STOCH(data_price['high'], data_price['low'], data_price['close'], fastk_period=9, slowk_period=5, slowk_matype=1, slowd_period=5, slowd_matype=1)
        data_price['kdj_k_ma5'] = ta.MA(data_price['kdj_k'], timeperiod=window_short, matype=0)
        data_price['slope'] = ta.LINEARREG_SLOPE(data_price['lma'], timeperiod=5)  # 5个周期的20日均线斜率

        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        # 均线策略的交易记录: 起始位置序号要把父类中的ATR计算周期也考虑在内
        start_id = max(1, window_long, self.ATR_Length)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond_1 = (data_price['sma'][i - 1] < data_price['lma'][i - 1]) and (data_price['sma'][i] > data_price['lma'][i])  # 金叉
            buy_cond_2 = (data_price['slope'][i] > 0) and (data_price['sma'][i - 1] > data_price['mma'][i - 1]) and (
                    data_price['sma'][i] < data_price['mma'][i]) and (data_price['sma'][i] > data_price['lma'][i])  # 长期均线向上，短期均线下穿了中期均线，短期均线在长期均线上方
            sell_cond_1 = (data_price['sma'][i - 1] > data_price['lma'][i - 1]) and (data_price['sma'][i] < data_price['lma'][i])  # 死叉

            # 情形一：当前无仓位且短均线上穿长均线(金叉)，则买入股票
            if (data_price['position'][i - 1] == 0) and buy_cond_1:
                self.set_position(data_price, i, "buy", '金叉买入')

            elif (data_price['position'][i - 1] == 0) and sell_cond_1:
                self.set_position(data_price, i, "sell", '死叉卖出')

            elif (data_price['position'][i - 1] == 0) and buy_cond_2:
                self.set_position(data_price, i, "buy", '多头死叉买入')

            # 情形二：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 情形三：当前持仓且短均线下穿长均线(死叉)，则卖出股票
            elif (data_price['position'][i - 1] == 1) and sell_cond_1:
                self.set_position(data_price, i, "buy_close", '死叉卖出平仓')

            elif (data_price['position'][i - 1] == -1) and buy_cond_1:
                self.set_position(data_price, i, "sell_close", '金叉买入平仓')

            # 情形五：高点回落大于阈值时平仓
            # 吊灯止盈法--吊灯的设置：固定/浮动
            # 作业3：开发N倍ATR作为高点回落出局条件
            elif (data_price['position'][i - 1] == 1) and ((max(data_price.high[i - 5:i]) - data_price['close'][i]) > self.ATR_times * data_price['atr'][i]):
                self.set_position(data_price, i, "buy_close", '{}回落平仓'.format(max(data_price.high[i - 5:i])))

            # 情形六：买入后三个bar还不赚钱立马平仓  计步器
            elif (data_price['position'][i - 1] == 1) and ((i - self.buy_entry_index) > pedometer) and (data_price['close'][i] / self.price_in < ratio):
                self.set_position(data_price, i, "buy_close", '{}周期未盈利平仓'.format(pedometer))

            # 情形七：盘中击穿重要支撑立马平仓
            # 作业4：把CDP指标写入到策略中，加入一个止损条件：当日价格如果击穿CDP的AL，也就是下方支撑，立即止损。
            # elif (data_price['position'][i - 1] == 1) and (data_price['close'][i] < data_price['al'][i]):
            #     self.set_position(data_price, i, "buy_close", '击穿CDP支撑{}止损平仓'.format(round(data_price['al'][i], 4)))

            # 情形八：逆势买入获得优势成本
            elif (data_price['position'][i - 1] == 0) and (data_price['kdj_k_ma5'][i] > 80) and (data_price['kdj_k'][i - 1] > 80) and (data_price['kdj_k'][i] < 80):
                self.set_position(data_price, i, "buy", '80上方回落买入')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        # 返回交易记录和全过程数据
        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 2. MACD
class MACD(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, fast=12, slow=26, sig_period=9, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['diff'], data_price['dea'], data_price['bar'] = ta.MACD(data_price['close'], fastperiod=fast, slowperiod=slow, signalperiod=sig_period)
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, slow)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['diff'][i] > 0) and (data_price['dea'][i] > 0) and (data_price['diff'][i - 1] < data_price['dea'][i - 1]) \
                       and (data_price['diff'][i] > data_price['dea'][i])  # 金叉
            sell_cond = (data_price['diff'][i] < 0) and (data_price['dea'][i] < 0) and (data_price['diff'][i - 1] > data_price['dea'][i - 1]) \
                        and (data_price['diff'][i] < data_price['dea'][i])  # 死叉

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '金叉买入')
            # 情形二：当前无持仓且死叉，则卖出
            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '死叉卖出')

            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '死叉卖出平仓')
                self.set_position(data_price, i, "sell", '死叉卖出(反手)')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '金叉买入平仓')
                self.set_position(data_price, i, "buy", '金叉买入(反手)')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 3. KDJ
class KDJ(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, fastk_period=9, slowk_period=5, slowd_period=5, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['kdj_k'], data_price['kdj_d'] = ta.STOCH(data_price['high'], data_price['low'], data_price['close'], fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=1, slowd_period=slowd_period, slowd_matype=1)
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, fastk_period)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['kdj_k'][i - 1] < data_price['kdj_d'][i - 1]) and (data_price['kdj_k'][i] > data_price['kdj_d'][i])  # 金叉
            sell_cond = (data_price['kdj_k'][i - 1] > data_price['kdj_d'][i - 1]) and (data_price['kdj_k'][i] < data_price['kdj_d'][i])  # 死叉

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '金叉买入')
            # 情形二：当前无持仓且死叉，则卖出
            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '死叉卖出')

            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '死叉卖出平仓')
                self.set_position(data_price, i, "sell", '死叉卖出(反手)')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '金叉买入平仓')
                self.set_position(data_price, i, "buy", '金叉买入(反手)')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 4. RSI
class RSI(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, timeperiod=6, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['rsi'] = ta.RSI(data_price['close'], timeperiod=timeperiod)
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, timeperiod)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['rsi'][i - 1] > 20) and (data_price['rsi'][i] < 20)
            sell_cond = (data_price['rsi'][i - 1] < 80) and (data_price['rsi'][i] > 80)

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '超卖反弹买入')

            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '超买反弹卖出')

            # 情形二：当前持仓且死叉，则卖出
            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '超买反弹卖出平仓')
                self.set_position(data_price, i, "sell", '超买反弹卖出(反手)')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '超卖反弹买入平仓')
                self.set_position(data_price, i, "buy", '超卖反弹买入(反手)')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 5. CCI
class CCI(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, period=14, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['cci'] = ta.CCI(data_price['high'], data_price['low'], data_price['close'], timeperiod=period)
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, period)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['cci'][i - 1] < 100) and (data_price['cci'][i] > 100)  # 从下向上突破100
            sell_cond = (data_price['cci'][i - 1] > -100) and (data_price['cci'][i] < -100)  # 从上向下突破-100

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '从下向上突破100买入')

            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '从上向下突破-100卖出')

            # 情形二：当前持仓且死叉，则卖出
            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '从上向下突破-100卖出平仓')
                self.set_position(data_price, i, "sell", '从上向下突破-100卖出(反手)')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '从下向上突破100买入平仓')
                self.set_position(data_price, i, "buy", '从下向上突破100买入(反手)')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 6. 布林线
class Bollinger(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, timeperiod=26, nbdevup=2, nbdevdn=2, matype=0, loss_ratio=0.10):
        '''
        :param timeperiod: 中轨均线的周期
        :param nbdevup: 上轨：中轨的几倍标准差，默认2
        :param nbdevdn: 下轨：中轨的几倍标准差，默认2
        :param matype: 均线的类型，默认简单移动平均
        :return:
        '''
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['upper'], data_price['middle'], data_price['lower'] = ta.BBANDS(data_price['close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, timeperiod)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['close'][i - 1] < data_price['upper'][i - 1]) and (data_price['close'][i] > data_price['upper'][i])  # 上穿上轨
            close_buy = (data_price['close'][i - 1] > data_price['middle'][i - 1]) and (data_price['close'][i] < data_price['middle'][i])  # 跌破中轨

            sell_cond = (data_price['close'][i - 1] > data_price['lower'][i - 1]) and (data_price['close'][i] < data_price['lower'][i])  # 下穿下轨
            close_sell = (data_price['close'][i - 1] < data_price['middle'][i - 1]) and (data_price['close'][i] > data_price['middle'][i])  # 突破中轨

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '上穿上轨')

            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '下穿下轨')

            elif (data_price['position'][i - 1] == 1) and close_buy:
                self.set_position(data_price, i, "buy_close", '跌破中轨卖出平仓')

            elif (data_price['position'][i - 1] == -1) and close_sell:
                self.set_position(data_price, i, "sell_close", '突破中轨买入平仓')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 7. SAR
class SAR(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    def run(self, acceleration=0.02, maximum=0.2, loss_ratio=0.10):
        '''
        默认10日抛物转向n=10
        :param acceleration: 步长
        :param maximum: 极值
        :return:
        '''
        data_price = self.data_price.copy()

        # 使用talib算法计算技术指标
        data_price['sar'] = ta.SAR(data_price['high'], data_price['low'], acceleration=acceleration, maximum=maximum)
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, 10)  # 10日抛物转向。不用talib的话可以改成n可变的情况，这里就是n
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['close'][i - 1] < data_price['sar'][i - 1]) and (data_price['close'][i] > data_price['sar'][i])  # 从下向上突破SAR
            sell_cond = (data_price['close'][i - 1] > data_price['sar'][i - 1]) and (data_price['close'][i] < data_price['sar'][i])  # 从上向下突破SAR

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '从下向上突破SAR买入')

            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '从上向下突破SAR卖出')

            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '从上向下突破SAR卖出平仓')
                self.set_position(data_price, i, "sell", '反手开空')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '从下向上突破SAR买入平仓')
                self.set_position(data_price, i, "buy", '反手开多')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 8. EXPMA
class EXPMA(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    @staticmethod
    def EXPMA(df, p1, p2):
        """
        指数加权移动平均线组合
        Args:
            p1 (int): 周期1
            p2 (int): 周期2
        """
        new_df = pd.DataFrame()
        new_df["ma1"] = df["close"].ewm(span=p1, adjust=False).mean()
        new_df["ma2"] = df["close"].ewm(span=p2, adjust=False).mean()
        return new_df

    def run(self, p1=5, p2=10, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 计算技术指标
        new = self.EXPMA(data_price, p1, p2)
        data_price['ma1'], data_price['ma2']= new["ma1"], new["ma2"]
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, p1, p2)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['ma1'][i - 1] < data_price['ma2'][i - 1]) and (data_price['ma1'][i] > data_price['ma2'][i])  # 金叉
            sell_cond = (data_price['ma1'][i - 1] > data_price['ma2'][i - 1]) and (data_price['ma1'][i] < data_price['ma2'][i])  # 死叉

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '金叉买入')

            if (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '死叉卖出')

            # 情形二：反手
            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '死叉卖出平仓')
                self.set_position(data_price, i, "sell", '死叉卖出(反手)')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '金叉买入平仓')
                self.set_position(data_price, i, "buy", '金叉买入(反手)')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 9. R-breaker 没指标了，搞两个策略
class R_Breaker(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super(R_Breaker, self).__init__(data_price, symbol, ATR_Length, ATR_times)

    @staticmethod
    def get_R_Breaker(df):
        new = CDP(df)
        new.rename(columns={'ah': 'b_break', 'al': 's_break', 'nh': 's_enter', 'nl': 'b_enter'}, inplace=True)
        return new

    def run(self, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 计算技术指标
        new = self.get_R_Breaker(data_price)
        data_price['b_break'], data_price['s_break'], data_price['s_enter'], data_price['b_enter'], data_price['s_setup'], data_price['b_setup'] = new["b_break"], new["s_break"], new["s_enter"], new["b_enter"], new["s_setup"], new["b_setup"]
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = 1
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = data_price['close'][i] > data_price['b_break'][i]  # 价格超过突破买入价
            sell_cond = data_price['close'][i] < data_price['s_break'][i]  # 价格跌破突破卖出价

            buy_backhand_cond = (data_price['high'][i] > data_price['s_setup'][i]) and (data_price['close'][i] < data_price['s_enter'][i])
            sell_backhand_cond = (data_price['low'][i] < data_price['b_setup'][i]) and (data_price['close'][i] > data_price['b_enter'][i])

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '价格超过突破买入价买入')

            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '价格跌破突破卖出价卖出')

            # 情形二：反手
            elif (data_price['position'][i - 1] == 1) and buy_backhand_cond:
                self.set_position(data_price, i, "buy_close", '最高价超过观察卖出价后跌破反转卖出价，卖出平仓')
                self.set_position(data_price, i, "sell", '反手开空')

            elif (data_price['position'][i - 1] == -1) and sell_backhand_cond:
                self.set_position(data_price, i, "sell_close", '最低价低于观察买入价后超过反转买入价，买入平仓')
                self.set_position(data_price, i, "buy", '反手开多')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 10. Donchian
class Donchian(BasicMethods):
    def __init__(self, data_price, symbol, ATR_Length, ATR_times):
        super().__init__(data_price, symbol, ATR_Length, ATR_times)

    @staticmethod
    def get_Donchian(df, n, m):
        new_df = pd.DataFrame()
        new_df["top"] = df["high"].shift(1).rolling(n).max()
        new_df["bottom"] = df["low"].shift(1).rolling(m).min()
        new_df["mid"] = 0.5 * (new_df["top"] + new_df["bottom"])
        return new_df

    def run(self, n=20, m=10, loss_ratio=0.10):
        data_price = self.data_price.copy()

        # 计算技术指标
        new = self.get_Donchian(data_price, n, m)
        data_price['top'], data_price['bottom'] = new["top"], new["bottom"]
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        start_id = max(1, n, m, self.ATR_Length)
        for i in range(start_id, data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond = (data_price['close'][i - 1] < data_price['top'][i - 1]) and (data_price['close'][i] > data_price['top'][i])  # 价格突破上轨
            sell_cond = (data_price['close'][i - 1] > data_price['bottom'][i - 1]) and (data_price['close'][i] < data_price['bottom'][i])  # 价格跌破下轨

            # 情形一：当前无仓位且金叉，则买入
            if (data_price['position'][i - 1] == 0) and buy_cond:
                self.set_position(data_price, i, "buy", '价格突破上轨买入')

            elif (data_price['position'][i - 1] == 0) and sell_cond:
                self.set_position(data_price, i, "sell", '价格跌破下轨卖出')

            elif (data_price['position'][i - 1] == -1) and buy_cond:
                self.set_position(data_price, i, "sell_close", '价格突破上轨买入平仓')
                self.set_position(data_price, i, "buy", '反手开多')

            elif (data_price['position'][i - 1] == 1) and sell_cond:
                self.set_position(data_price, i, "buy_close", '价格跌破下轨卖出平仓')
                self.set_position(data_price, i, "sell", '反手开空')

            # 情形三：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "buy_close", '多头止损平仓')

            elif (data_price['position'][i - 1] == -1) and ((data_price['close'][i] / self.price_out - 1.0) > loss_ratio):
                self.set_position(data_price, i, "sell_close", '空头止损平仓')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        transactions_buy, transactions_sell, data_price = self.get_results(data_price, start_id)
        return transactions_buy, transactions_sell, data_price


# 执行一个品种的所有策略
def Backtesting(file_path, file_name, period='day', ATR_Length=20, ATR_times=2, market_code=0, real_time_stock=False):
    # 数据获取和处理，得到strategies用的data
    data, N = pre_process(file_path, file_name, period, ATR_Length, market_code, real_time_stock)
    # 实例化策略类
    s1 = DualMovingAverage(data, file_name, ATR_Length, ATR_times)
    s2 = MACD(data, file_name, ATR_Length, ATR_times)
    s3 = KDJ(data, file_name, ATR_Length, ATR_times)
    s4 = RSI(data, file_name, ATR_Length, ATR_times)
    s5 = CCI(data, file_name, ATR_Length, ATR_times)
    s6 = Bollinger(data, file_name, ATR_Length, ATR_times)
    s7 = SAR(data, file_name, ATR_Length, ATR_times)
    s8 = EXPMA(data, file_name, ATR_Length, ATR_times)
    s9 = R_Breaker(data, file_name, ATR_Length, ATR_times)
    s10 = Donchian(data, file_name, ATR_Length, ATR_times)

    strategy_list = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    print('------------------------------\n品种名称: {}\n'.format(file_name))
    record = []
    for strategy in strategy_list:
        stra_name = strategy.__class__.__name__
        print('------------------------------\n策略名称: {}\n'.format(stra_name))
        transactions_buy, transactions_sell, data_price = strategy.run()  # 运行某个策略，返回结果
        Sharpe = show_performance(stra_name, transactions_buy, transactions_sell, data_price, N, file_path)  # show该策略评价指标，策略净值可视化
        record.append(Sharpe)
    return record


if __name__ == "__main__":
    path = 'E:/PycharmProjects/Quant/Course-Kang/Lesson_2_rule/作业/导出数据/'
    symbols = ["PTA", "白糖SR", "豆粕M", "豆油Y", "沪铜CU", "沪锌ZN", "螺纹钢RB", "塑料L", "铁矿石I", "橡胶RU"]
    strategy_name = ['DualMovingAverage', 'MACD', 'KDJ', 'RSI', 'CCI', 'Bollinger', 'SAR', 'EXPMA', 'R_Breaker', 'Donchian']
    sharpe_record = pd.DataFrame(index=symbols, columns=strategy_name)
    for s in symbols:
        record_row = Backtesting(path, s)
        sharpe_record.loc[s] = record_row
        sharpe_record.to_csv(path + 'sharpe_record.csv', encoding='utf_8_sig')  # encoding='utf_8_sig'防止中文乱码   每测一个品种就刷新一次文件，防止中间断了啥也没了
