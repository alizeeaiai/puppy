from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import talib as ta
import time
import datetime
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import glob

def dataclean(data):
    data = data[["trade_date", "open", "high", "low", "close", "volume"]]
    data.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    data = data.set_index("timestamp")
    data = data.replace([np.nan, np.inf, -np.inf], 0.0)
    return data


# CH:=REF(H,1);
# CL:=REF(L,1);
# CC:=REF(C,1);
# CDP:(CH+CL+CC)/3;
# AH:2*CDP+CH-2*CL;
# NH:CDP+CDP-CL;
# NL:CDP+CDP-CH;
# AL:2*CDP-2*CH+CL;
class Strategies:
    def __init__(
        self,
        data_price,
        window_short=5,
        window_median=10,
        window_long=20,
        loss_ratio=0.20,
    ):

        self.data_price = data_price
        self.window_short = window_short
        self.window_median = window_median
        self.window_long = window_long
        self.loss_ratio = loss_ratio
        self.indicdf = self.indic

    @property
    def indic(self):

        data = self.data_price # data就是rawdata
        # print(type(data.index))
        data_price = data.copy()
        # data_price.index = data_price.index.strftime('%Y%m%d')
        # 使用talib算法计算技术指标
        data_price["sma"] = ta.MA(
            data_price["close"], timeperiod=self.window_short, matype=0
        )
        data_price["lma"] = ta.MA(
            data_price["close"], timeperiod=self.window_long, matype=0
        )
        data_price["mma"] = ta.MA(
            data_price["close"], timeperiod=self.window_median, matype=0
        )
        data_price["kdj_k"], data_price["kdj_d"] = ta.STOCH(
            data_price["high"],
            data_price["low"],
            data_price["close"],
            fastk_period=9,
            slowk_period=5,
            slowk_matype=1,
            slowd_period=5,
            slowd_matype=1,
        )
        data_price["kdj_k_ma5"] = ta.MA(
            data_price["kdj_k"], timeperiod=self.window_short, matype=0
        )
        data_price["lslope"] = ta.LINEARREG_SLOPE(data_price.lma, 5)
        data_b1 = data[["high", "low", "close"]]
        data_b1["CDP"] = np.mean(data_b1[["high", "low", "close"]], 1) # np.mean(axis=1)计算每一行的均值
        data_b1["AL"] = 2 * data_b1.CDP - data_b1.high - data_b1.low
        data_price["AL"] = data_b1["AL"] # 作业三的CDP的值
        data_price["ATR"] = ta.ATR(
            data_b1.high, data_b1.low, data_b1.close, timeperiod=3
        )

        ema12 = data_price["close"].ewm(span=12, adjust=False).mean()
        ema26 = data_price["close"].ewm(span=26, adjust=False).mean()
        # 计算DIF和DEA
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        # 计算MACD
        data_price["macd"] = (dif - dea) * 2

        delta = data_price["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().abs()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data_price["rsi_diff"] = rsi - rsi.shift(1)

        upper_2sd, mid_2sd, lower_2sd = ta.BBANDS(
            data_price["close"], nbdevup=2, nbdevdn=2, timeperiod=20
        )
        data_price["ADX"] = ta.ADX(
            data["high"], data["low"], data["close"], timeperiod=14
        )
        data_price["DI+"] = ta.PLUS_DI(
            data["high"], data["low"], data["close"], timeperiod=14
        )
        data_price["DI-"] = ta.MINUS_DI(
            data["high"], data["low"], data["close"], timeperiod=14
        )
        data_price["MTM"] = ta.MOM(data["close"], timeperiod=10)
        data_price["MTM_signal"] = ta.MA(data_price["MTM"], timeperiod=10)
        # 计算20日和50日移动平均线
        data["MA20"] = data["close"].rolling(window=20).mean()
        data["MA50"] = data["close"].rolling(window=50).mean()

        # 计算趋势线指标
        data_price["TL"] = np.where(data["MA20"] > data["MA50"], 1, -1)

        # data['CDP'] = (data['high'] + data['low'] + data['close']) / 3
        # data['AH'] = data['CDP'] + (data['high'] - data['low'])
        # data['AL'] = data['CDP'] - (data['high'] - data['low'])
        # data['NH'] = 2 * data['CDP'] - data['low']
        # data['NL'] = 2 * data['CDP'] - data['high']

        # print(data_b1)

        return data_price

    def HLC(self):
        # 计算HLC指标
        data = self.indicdf
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        data["hlc"] = ta.CMO(close, timeperiod=14)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["hlc"] > data["hlc"].shift(1)) & (data["hlc"] < 0), 1, 0
        )
        data["sell_signal"] = np.where(
            (data["hlc"] < data["hlc"].shift(1)) & (data["hlc"] > 0), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def ROC_Strategy(self):
        # 计算ROC指标
        data = self.indicdf
        close = data["close"].values
        data["roc"] = ta.ROCR(close, timeperiod=10)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["roc"] > data["roc"].shift(1)) & (data["roc"] > 1), 1, 0
        )
        data["sell_signal"] = np.where(
            (data["roc"] < data["roc"].shift(1)) & (data["roc"] > 1), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def KDJ_Strategy(self):
        # 计算KDJ指标
        data = self.indicdf
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        data["slowk"], data["slowd"] = ta.STOCH(high, low, close)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["slowk"] > data["slowd"])
            & (data["slowk"].shift(1) < data["slowd"].shift(1)),
            1,
            0,
        )
        data["sell_signal"] = np.where(
            (data["slowk"] < data["slowd"])
            & (data["slowk"].shift(1) > data["slowd"].shift(1)),
            1,
            0,
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def ADX(self):
        data = self.indicdf
        # 计算交易信号
        data["signal"] = 0
        data.loc[
            (data["DI+"] > data["ADX"]) & (data["DI+"].shift(1) < data["ADX"].shift(1)),
            "signal",
        ] = 1
        data.loc[
            (data["DI-"] > data["ADX"]) & (data["DI-"].shift(1) < data["ADX"].shift(1)),
            "signal",
        ] = -1

        # 计算持仓状态
        data["position"] = data["signal"].shift(1)

        # 生成交易信号
        data["trading_signal"] = 0
        data.loc[
            (data["position"] == 1) & (data["position"].shift(1) == -1),
            "trading_signal",
        ] = 1  # 金叉买入
        data.loc[
            (data["position"] == -1) & (data["position"].shift(1) == 1),
            "47trading_signal",
        ] = -1  # 死叉卖出

        # 计算收益
        data["pnl"] = data["trading_signal"] * data["close"].pct_change()

        # 计算累计收益率
        data["cumulative_return"] = (1 + data["pnl"]).cumprod()
        return data

    def ADX_Strategy(self):
        # 计算ADX指标
        data = self.indicdf
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        data["adx"] = ta.ADX(high, low, close)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["adx"] > data["adx"].shift(1)) & (data["adx"] > 20), 1, 0
        )
        data["sell_signal"] = np.where(
            (data["adx"] < data["adx"].shift(1)) & (data["adx"] > 20), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def MTM(self):
        # 计算交易信号
        data = self.indicdf
        data["signal"] = 0
        data.loc[
            (data["MTM"] > data["MTM_signal"])
            & (data["MTM"].shift(1) < data["MTM_signal"].shift(1)),
            "signal",
        ] = 1
        data.loc[
            (data["MTM"] < data["MTM_signal"])
            & (data["MTM"].shift(1) > data["MTM_signal"].shift(1)),
            "signal",
        ] = -1

        # 计算持仓状态
        data["position"] = data["signal"].shift(1)

        # 生成交易信号
        data["trading_signal"] = 0
        data.loc[
            (data["position"] == 1) & (data["position"].shift(1) == -1),
            "trading_signal",
        ] = 1  # 金叉买入
        data.loc[
            (data["position"] == -1) & (data["position"].shift(1) == 1),
            "trading_signal",
        ] = -1  # 死叉卖出

        # 计算收益
        data["pnl"] = data["trading_signal"] * data["close"].pct_change()

        # 计算累计收益率
        data["cumulative_return"] = (1 + data["pnl"]).cumprod()
        return data

    def MTM_Strategy(self):
        # 计算MTM指标
        data = self.indicdf
        close = data["close"].values
        data["mtm"] = ta.MOM(close, timeperiod=10)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["mtm"] > data["mtm"].shift(1)) & (data["mtm"] < 0), 1, 0
        )
        data["sell_signal"] = np.where(
            (data["mtm"] < data["mtm"].shift(1)) & (data["mtm"] > 0), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def MA_hwpart1(
        self, window_short=5, window_median=10, window_long=20, loss_ratio=0.20
    ):
        # df_price:价格数据；
        # window_short：短均线周期，默认为5；
        # window_long：长均线周期，默认为10；
        # lossratio：止损率,默认为1%，即开仓后下跌超过1%止损。
        # 2.0 get基础数据并计算
        data_price = self.indicdf # indicdf就是raw数据

        data_price["position"] = 0.0  # 记录仓位
        data_price["flag"] = 0.0  # 记录买卖
        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        ##2.1绘制K线和均线

        # self.plotma(data_price)

        # 2.2均线策略的交易记录
        Buy = []  # 保存买入记录
        Sell = []  # 保存卖出记录
        price_in = 1  # 初始买入价设置为1
        data_price["position"] = 0

        for i in range(max(1, window_long), data_price.shape[0]):
            data_price = data_price.copy()

            # todo:buy sell 函数封装
            buy_cond_1 = (data_price["sma"][i - 1] < data_price["lma"][i - 1]) and (
                data_price["sma"][i] > data_price["lma"][i]
            )
            buy_cond_2 = (
                (data_price["lma"][i] > data_price["lma"][i - 1])
                and (data_price["sma"][i - 1] > data_price["mma"][i - 1])
                and (data_price["sma"][i] < data_price["mma"][i])
                and (data_price["sma"][i] > data_price["lma"][i])
            )
            # if buy_cond_2:
            #     print(data_price.index[i], '=====condtion_2=========')

            # print(data_price.index[i], '============', max(data_price.high[i-5:i]))

            # 情形一：当前无仓位且短均线上穿长均线(金叉)，则买入股票 buy_cond_1
            if (
                (data_price["position"][i - 1] == 0)
                and (data_price["sma"][i - 1] < data_price["lma"][i - 1])
                and (data_price["sma"][i] > data_price["lma"][i])
            ):

                data_price["flag"][i] = 1  # 记录做多还是做空，这里1是做多
                data_price["position"][i] = 1  # 仓位记录为1，表示有1手仓位
                date_in = data_price.index[i]  # 记录买入的时间 年-月-日
                price_in = data_price["close"][i]  # 记录买入的价格，这里是以收盘价买入
                entry_index = i
                # print(data_price.index[i], '=========金叉买入@--', price_in)
                Buy.append([date_in, price_in, "金叉买入"])  # 把买入记录保存到Buy列表里
                # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

            elif (data_price["position"][i - 1] == 0) and buy_cond_2:
                # print(data_price.index[i], '=========多头死叉买入')
                data_price["flag"][i] = 1  # 记录做多还是做空，这里1是做多
                data_price["position"][i] = 1  # 仓位记录为1，表示有1手仓位
                date_in = data_price.index[i]  # 记录买入的时间 年-月-日
                price_in = data_price["close"][i]  # 记录买入的价格，这里是以收盘价买入
                entry_index = i
                Buy.append([date_in, price_in, "多头死叉买入"])  # 把买入记录保存到Buy列表里
                # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

            # 情形二：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price["position"][i - 1] == 1) and (
                (1.0 - data_price["close"][i] / price_in) > loss_ratio
            ):
                print('这里出事故了')
                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                Sell.append([date_out, price_out, "止损平仓"])  # 把卖出记录保存到Sell列表里
                # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

            # 情形三：当前持仓且短均线下穿长均线(死叉)，则卖出股票
            elif (
                (data_price["position"][i - 1] == 1)
                & (data_price["sma"][i - 1] > data_price["lma"][i - 1])
                & (data_price["sma"][i] < data_price["lma"][i])
            ):
                # print(data_price.index[i], '=========死叉卖出')
                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                Sell.append([date_out, price_out, "死叉卖出"])  # 把卖出记录保存到Sell列表里

            # 情形四：当前持仓且短均线下穿长均线(死叉)，则卖出股票
            elif (
                (data_price["position"][i - 1] == 1)
                & (data_price["sma"][i - 1] > data_price["lma"][i - 1])
                & (data_price["sma"][i] < data_price["lma"][i])
            ):
                # print(data_price.index[i], '=========死叉卖出')
                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                Sell.append([date_out, price_out, "死叉卖出"])  # 把卖出记录保存到Sell列表里

            # 情形六：买入后三个bar还不赚钱立马平仓
            elif (
                (data_price["position"][i - 1] == 1)
                and (i - entry_index > 2)
                and (data_price["close"][i] / price_in < 0.985)
            ):
                # print(i - entry_index, ' hodling for a period')
                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                # print(data_price.index[i], '============', '=========持仓超过3个周期还未盈利达到1%平仓@--', price_out)
                Sell.append([date_out, price_out, "死叉平仓"])  # 把卖出记录保存到Sell列表里

            # 情形七：盘中击穿重要支撑立马平仓

            # 情形八：逆势买入获得优势成本
            elif (
                (data_price["position"][i - 1] == 0)
                and (data_price["kdj_k_ma5"][i] > 80)
                and (data_price["kdj_k"][i - 1] > 80)
                and (data_price["kdj_k"][i] < 80)
            ):
                data_price["flag"][i] = 1  # 记录做多还是做空，这里1是做多
                data_price["position"][i] = 1  # 仓位记录为1，表示有1手仓位
                date_in = data_price.index[i]  # 记录买入的时间 年-月-日
                price_in = data_price["close"][i]  # 记录买入的价格，这里是以收盘价买入
                entry_index = i
                # print(data_price.index[i], "=========80上方回落买入@--", price_in)
                Buy.append([date_in, price_in, "80上方回落买入"])  # 把买入记录保存到Buy列表里

            # 海龟交易法则按照价格高于初始价格0.5ATR进行加仓操作,按照价格低于建仓价2ATR进行止损操作.
            elif (data_price["position"][i - 1] == 0) and (
                data_price["close"][i - 1]
                >= data_price["open"][0] + 0.5 * data_price["ATR"][i - 1]
            ):

                data_price["flag"][i] = 1  # 记录做多还是做空，这里1是做多
                data_price["position"][i] = 1  # 仓位记录为1，表示有1手仓位
                date_in = data_price.index[i]  # 记录买入的时间 年-月-日
                price_in = data_price["close"][i]  # 记录买入的价格，这里是以收盘价买入
                entry_index = i
                Buy.append([date_in, price_in, "0.5ATR买入"])  # 把买入记录保存到Buy列表里
                # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

            # 按照价格低于建仓价2ATR进行止损操作 海龟的止损设置在买入价格以下的2ATR
            elif (data_price["position"][i - 1] == 1) and (
                data_price["close"][i - 1]
                <= data_price["close"][entry_index] - 2 * data_price["ATR"][i - 1]
            ):

                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                Sell.append([date_out, price_out, "2倍ATR卖出"])  # 把卖出记录保存到Sell列表里

            # 把CDP指标写入到策略中，加入一个止损条件：当日价格如果击穿CDP的AL，也就是下方支撑，立即止损
            elif (data_price["position"][i - 1] == 1) and (
                data_price["low"][i - 1] <= data_price["AL"][i - 1]
            ):

                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                Sell.append([date_out, price_out, "击穿AL卖出"])  # 把卖出记录保存到Sell列表里

            # 当股票价格均线斜率超过45度以上向上运行时，股票价格K线上穿均线，投资者可以作为参考买点信号之一。
            elif (
                (data_price["position"][i - 1] == 0)
                and (data_price["lslope"][i - 1] > 1)
                and (data_price["close"][i - 1] > data_price["lma"][i - 1])
            ):

                data_price["flag"][i] = 1  # 记录做多还是做空，这里1是做多
                data_price["position"][i] = 1  # 仓位记录为1，表示有1手仓位
                date_in = data_price.index[i]  # 记录买入的时间 年-月-日
                price_in = data_price["close"][i]  # 记录买入的价格，这里是以收盘价买入
                entry_index = i
                # print(data_price.index[i], '=========金叉买入@--', price_in)
                Buy.append([date_in, price_in, "slope买入"])  # 把买入记录保存到Buy列表里

            # 当股票价格均线斜率超过45度以上向下运行时，股票价格K线下穿均线，投资者可以作为参考卖点信号之一。
            elif (
                (data_price["position"][i - 1] == 0)
                and (data_price["lslope"][i - 1] <= -1)
                and (data_price["close"][i - 1] < data_price["lma"][i - 1])
            ):

                data_price["flag"][i] = -1  # 记录做多还是做空，这里-1是做多
                data_price["position"][i] = 0  # 仓位记录为0，表示没有仓位了
                date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
                price_out = data_price["close"][i]  # 记录卖出的价格，这里是以收盘价卖出
                Sell.append([date_out, price_out, "slope止损"])  # 把卖出记录保存到Sell列表里
            else:
                data_price["position"][i] = data_price["position"][i - 1]

            # print(data_price.index[i], '======================', data_price['position'][i])
        # print(data_price.tail(60))
        p1 = pd.DataFrame(Buy, columns=["买入日期", "买入价格", "备注"])
        p2 = pd.DataFrame(Sell, columns=["卖出日期", "卖出价格", "备注"])
        transactions = pd.concat([p1, p2], axis=1)  # 交易记录

        data_price = data_price.iloc[window_long:, :]
        data_price["ret"] = data_price.close.pct_change(1).fillna(0).shift(-1)

        data_price["nav"] = (1 + data_price.ret * data_price.position).cumprod()
        data_price["benchmark"] = data_price.close / data_price.close[0]


        ##2.3返回交易记录和全过程数据
        return transactions, data_price

    def RSI_Strategy(self):
        # 计算RSI指标
        data = self.indicdf
        close = data["close"].values
        data["rsi"] = ta.RSI(close)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["rsi"] > 50) & (data["rsi"].shift(1) <= 50), 1, 0
        )
        data["sell_signal"] = np.where(
            (data["rsi"] < 50) & (data["rsi"].shift(1) >= 50), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def TL_Strategy(self):
        # 计算趋势线指标
        data = self.indicdf
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        data["tl"] = ta.HT_TRENDLINE(close)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["close"] > data["tl"])
            & (data["close"].shift(1) < data["tl"].shift(1)),
            1,
            0,
        )
        data["sell_signal"] = np.where(
            (data["close"] < data["tl"])
            & (data["close"].shift(1) > data["tl"].shift(1)),
            1,
            0,
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def EMA_Strategy(self):
        # 计算EMA指标
        data = self.indicdf
        close = data["close"].values
        data["ema5"] = ta.EMA(close, timeperiod=5)
        data["ema10"] = ta.EMA(close, timeperiod=10)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["ema5"] > data["ema10"])
            & (data["ema5"].shift(1) <= data["ema10"].shift(1)),
            1,
            0,
        )
        data["sell_signal"] = np.where(
            (data["ema5"] < data["ema10"])
            & (data["ema5"].shift(1) >= data["ema10"].shift(1)),
            1,
            0,
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def TRIX_Strategy(self):
        # 计算TRIX指标
        data = self.indicdf
        close = data["close"].values
        data["trix"] = ta.TRIX(close)

        # 生成交易信号
        data["buy_signal"] = np.where(
            (data["trix"] > data["trix"].shift(1)) & (data["trix"].shift(1) < 0), 1, 0
        )
        data["sell_signal"] = np.where(
            (data["trix"] < data["trix"].shift(1)) & (data["trix"].shift(1) > 0), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def SAR_Strategy(self):
        # 计算SAR指标
        data = self.indicdf
        high = data["high"].values
        low = data["low"].values
        data["sar"] = ta.SAR(high, low)

        # 生成交易信号
        data["buy_signal"] = np.where(data["close"] > data["sar"], 1, 0)
        data["sell_signal"] = np.where(data["close"] < data["sar"], 1, 0)


        rtdict = self.signal_to_res(data)
        return rtdict

    def BOLL_Strategy(self):
        # 计算BOLL指标
        data = self.indicdf
        close = data["close"].values
        upper, middle, lower = ta.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        data["upper"] = upper
        data["middle"] = middle
        data["lower"] = lower

        # 生成交易信号
        data["buy_signal"] = np.where(
            (close > data["upper"].shift(1)) & (close < data["upper"]), 1, 0
        )
        data["sell_signal"] = np.where(
            (close < data["lower"].shift(1)) & (close > data["lower"]), 1, 0
        )


        rtdict = self.signal_to_res(data)
        return rtdict

    def MESA_Strategy(self):
        # 计算MESA指标
        data = self.indicdf
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        mesa = ta.MAMA(close, fastlimit=0.5, slowlimit=0.05)[0]
        mesa = pd.Series(mesa)
        # 生成交易信号
        data["buy_signal"] = np.where((mesa > mesa.shift(1)) & (mesa < close), 1, 0)
        data["sell_signal"] = np.where((mesa < mesa.shift(1)) & (mesa > close), 1, 0)
        data["mesa"] = mesa

        rtdict = self.signal_to_res(data)
        return rtdict

    def plotma(self):
        data_price = self.indicdf
        kline = Kline(
            init_opts=opts.InitOpts(
                width="1200px", height="600px", theme=ThemeType.DARK
            )
        )  # 设置K线图的长和宽
        kline.add_xaxis(data_price.index.tolist())  # 将index也就是时间轴设置为X轴
        y = list(
            data_price.loc[:, ["open", "high", "low", "close"]].round(4).values
        )  # 设置为list，一共有data_price.shape[0]个，等待传入Kbar
        y = [i.tolist() for i in y]  # 里面的单个数组也必须转换成list
        kline.add_yaxis("Kline", y)
        # kline.extend_axis(yaxis=opts.AxisOpts( axislabel_opts=opts.LabelOpts(formatter="{value}") ))
        kline.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
        kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],  # 内部滑动
            title_opts=opts.TitleOpts(
                title="510050_50ETF_Kline_and_MA", pos_left="45%"
            ),  # 题目位置
            legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", axis_pointer_type="cross"
            ),  # 添加趋势线
        )

        line = Line()
        line.add_xaxis(data_price.index.tolist())
        line.add_yaxis("MA_short", data_price.sma.round(2).tolist(), is_smooth=True)
        line.add_yaxis("MA_median", data_price.mma.round(2).tolist(), is_smooth=True)
        line.add_yaxis("MA_long", data_price.lma.round(2).tolist(), is_smooth=True)
        line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
        line.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", axis_pointer_type="cross"
            ),  # 添加趋势线
        )
        kline.overlap(line)
        kline.render("510050_50ETF_Kline_and_MA.html")

    def signal_to_res(self, data):

        data["position"] = np.where(data["buy_signal"] == 1, 1, np.nan)
        data["position"] = np.where(data["sell_signal"] == 1, 0, data["position"])
        data["position"].fillna(method="ffill", inplace=True)
        # 计算每日收益率
        data["returns"] = data["close"].pct_change() * data["position"].shift(1)
        data["cum_returns"] = (1 + data["returns"]).cumprod()

        annualized_return = (1 + data["cum_returns"].iloc[-1]) ** (252 / len(data)) - 1
        annualized_volatility = data["returns"].std() * (252**0.5)
        sharpe_ratio = annualized_return / annualized_volatility
        rtdict = {"cum_returns": data["cum_returns"][-1], "sharpe_ratio": sharpe_ratio}
        return rtdict


if __name__ == "__main__":
    rawdata = pd.read_excel("/Users/aiailyu/PycharmProjects/第3节课/1_510050_d.xlsx")
    sobj = Strategies(rawdata)
    sh_data = sobj.MA_hwpart1()
    print("hw_part1:\n", sh_data)

    # todo:1更改輸入数据格式 2遍历所有策略

    files = glob.glob("Commodities_data/*.csv") # 遍历csv下所有的文件

    reslist = []
    for f in files:
        df = pd.DataFrame()
        df = pd.read_csv(f)
        df = dataclean(df)
        # df = df.append(csv)
        # print(df.iloc[-1])
        sobj = Strategies(df)
        # data = sobj.MESA_Strategy()
        attrs = (getattr(sobj, name) for name in dir(sobj))
        # methods = [attrs]
        stra_list = [
            func
            for func in dir(sobj)
            if callable(getattr(sobj, func)) and func.endswith("Strategy")
        ]

        for stra in stra_list:
            # stradic = {'strategy', stra}
            rsdict = getattr(sobj, stra)()

            rsdict["strategy"] = stra

            comd = f.replace("Commodities_data\\", "").replace(".csv", "")

            rsdict["comd"] = comd

            reslist.append(rsdict)
        # print(reslist)

    resdf = pd.DataFrame(reslist)

    print("\n")
    print("hw_part2:\n", resdf)
