import numpy as np
import pandas as pd
import time
import datetime
import talib as ta
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import warnings
warnings.filterwarnings('ignore')


'''
说明：
作业2涉及131、141行；作业3涉及87、163行；作业4涉及47、88、172行。
'''


class Strategy:
    def __init__(self, file_path, file_name, ATR_Length, ATR_times, period, stock=True):
        '''
        :param file_path: 文件夹路径
        :param file_name: 品种文件名。用实时数据接口的品种代码命名文件，与实时数据获取共用名称
        :param ATR_Length: ATR的计算周期
        :param ATR_times: ATR止损的乘数(倍数)
        :param period: K线周期。'5min', '15min', '30min', 'h', 'day', '1min'  见self.Dic对照表
        :param stock: 是否是股票
        '''
        self.file_path = file_path
        self.symbol = file_name
        # 按股票计算的对应字典：{周期: (tdx的K线周期代号, 一年有多少周期)}  期货的可以再修改
        self.Dic = {'5min': (0, 12000), '15min': (1, 4000), '30min': (2, 2000), 'h': (3, 1000), 'day': (4, 250), '1min': (7, 60000)}
        self.period = period
        self.N = self.Dic[period][1]  # 一年有多少个周期，这里取出来是250
        self.ATR_Length = ATR_Length
        self.ATR_times = ATR_times
        self.stock = stock  # 若测试品种是股票，就用股票的实时数据接口
        self.Buy = []  # 保存买入记录
        self.Sell = []  # 保存卖出记录
        self.price_in = 1  # 初始买入价设置为1
        self.entry_index = 0
        self.data_price = pd.read_csv(file_path + file_name + '.csv', usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    @staticmethod
    def CDP(df):
        """
        逆势操作指标
        """
        new_df = pd.DataFrame()
        ch = df["high"].shift(1)
        cl = df["low"].shift(1)
        cc = df["close"].shift(1)
        cdp = (ch + cl + cc) / 3
        new_df["ah"] = 2 * cdp + ch - 2 * cl
        new_df["al"] = 2 * cdp - 2 * ch + cl
        new_df["nh"] = 2 * cdp - cl
        new_df["nl"] = 2 * cdp - ch
        return new_df

    def data_processing(self):
        # 这一步做完，历史数据和实时数据的拼接就完成了
        if self.stock:  # 如果是股票，调用tdx实时接口
            api = TdxHq_API()
            if api.connect('119.147.212.81', 7709):  # 注意这里的IP地址和数据接口
                current_data = api.to_df(api.get_security_bars(self.Dic[self.period][0], 1, self.symbol, 1, 200))
                api.disconnect()  # 调用完以后一定要关闭接口

            current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
            current_data.rename(columns={'datetime': 'timestamp', 'vol': 'volume'}, inplace=True)
            self.data_price = pd.concat([self.data_price, current_data], axis=0)  # 合并数据

        if self.period == 'day':
            self.data_price['timestamp'] = pd.to_datetime(self.data_price['timestamp']).dt.date

        self.data_price['timestamp'] = pd.to_datetime(self.data_price['timestamp'])
        self.data_price = self.data_price.sort_values(by='timestamp', ascending=True)
        self.data_price = self.data_price.drop_duplicates('timestamp')  # 排序并去重
        self.data_price = self.data_price.set_index('timestamp')  # 时间设为索引
        self.data_price['year'] = self.data_price.index.year  # 增加一列年份，用于策略评价
        self.data_price = self.data_price.replace([np.inf, -np.inf, np.nan], 0.0)  # 异常值清理。也可不加，在具体的strategy中会清理

        # 增加列。atr和al为所有strategy通用列
        self.data_price['position'] = 0.0  # 记录仓位
        self.data_price['flag'] = 0.0  # 记录买卖，1为做多，-1为平仓
        self.data_price['atr'] = ta.ATR(self.data_price['high'], self.data_price['low'], self.data_price['close'], self.ATR_Length)  # 计算self.ATR_Length周期的ATR，self.ATR_Lenghth为atr中timeperiod的参数设置，这里是20
        self.data_price['al'] = self.CDP(self.data_price)['al']  # CDP的支撑位

    def set_position(self, data_price, i, direction, remarks):
        '''
        只做多、平多版本（不做空）
        :param i: 第i根K线
        :param remarks: 交易备注
        :param direction: 交易方向，"buy", "close"
        :return:
        '''
        if direction == "buy":
            data_price['flag'][i] = 1  # 开多
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年月日
            self.price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            self.entry_index = i  # 做多时，用全局变量记录这根序号
            self.Buy.append([date_in, self.price_in, remarks]) # 把买入记录保存到Buy列表里
            print(date_in, '========={}--'.format(remarks), self.price_in)
        elif direction == "close":
            data_price['flag'][i] = -1  # 平仓
            data_price['position'][i] = 0
            date_out = data_price.index[i]
            price_out = data_price['close'][i]
            self.Sell.append([date_out, price_out, remarks])
            print(date_out, '========={}--'.format(remarks), price_out)
        else:
            raise ValueError("交易方向只能为'buy'或'close'")

    def Dual_Moving_Average(self, window_short=5, window_median=10, window_long=20, loss_ratio=0.20):  # data_price,
        # data_price_price:价格数据
        # window_short：短均线周期，默认为5；
        # window_long：长均线周期，默认为10；
        # loss_ratio：止损率,默认为1%，即开仓后下跌超过1%止损。
        self.data_processing()
        data_price = self.data_price.copy()
        data_price.index = data_price.index.strftime('%Y%m%d')  # strftime把日期解析成字符串格式

        # 使用talib算法计算技术指标
        data_price['sma'] = ta.MA(data_price['close'], timeperiod=window_short, matype=0)
        data_price['lma'] = ta.MA(data_price['close'], timeperiod=window_long, matype=0)
        data_price['mma'] = ta.MA(data_price['close'], timeperiod=window_median, matype=0)
        data_price['kdj_k'], data_price['kdj_d'] = ta.STOCH(data_price['high'], data_price['low'], data_price['close'], fastk_period=9, slowk_period=5, slowk_matype=1, slowd_period=5, slowd_matype=1)
        data_price['kdj_k_ma5'] = ta.MA(data_price['kdj_k'], timeperiod=window_short, matype=0)
        data_price['slope'] = ta.LINEARREG_SLOPE(data_price['lma'], timeperiod=5)  # 5个周期的20日均线斜率

        data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)

        # 2.2均线策略的交易记录
        for i in range(max(1, window_long), data_price.shape[0]):
            data_price = data_price.copy()

            buy_cond_1 = (data_price['sma'][i - 1] < data_price['lma'][i - 1]) and (data_price['sma'][i] > data_price['lma'][i])  # 金叉
            # 作业2：用slope来表示5个周期的20日均线向上。用(data_price['slope'][i] > 0)代替了条件(data_price['lma'][i] > data_price['lma'][i - 1])
            buy_cond_2 = (data_price['slope'][i] > 0) and (data_price['sma'][i - 1] > data_price['mma'][i - 1]) and (
                    data_price['sma'][i] < data_price['mma'][i]) and (data_price['sma'][i] > data_price['lma'][i])  # 长期均线向上，短期均线下穿了中期均线，短期均线在长期均线上方

            # 情形一：当前无仓位且短均线上穿长均线(金叉)，则买入股票
            if (data_price['position'][i - 1] == 0) and buy_cond_1:
                self.set_position(data_price, i, "buy", '金叉买入')

            elif (data_price['position'][i - 1] == 0) and buy_cond_2:
                self.set_position(data_price, i, "buy", '多头死叉买入')

            # 情形二：当前持仓且下跌超过止损率，则平仓止损
            elif (data_price['position'][i - 1] == 1) and ((1.0 - data_price['close'][i] / self.price_in) > loss_ratio):
                self.set_position(data_price, i, "close", '止损平仓')

            # 情形三：当前持仓且短均线下穿长均线(死叉)，则卖出股票
            elif (data_price['position'][i - 1] == 1) & (data_price['sma'][i - 1] > data_price['lma'][i - 1]) & (
                    data_price['sma'][i] < data_price['lma'][i]):
                self.set_position(data_price, i, "close", '死叉卖出')

            # 情形五：高点回落大于阈值时平仓
            # 吊灯止盈法--吊灯的设置：固定/浮动
            # 作业3：开发N倍ATR作为高点回落出局条件
            elif (data_price['position'][i - 1] == 1) and ((max(data_price.high[i - 5:i]) - data_price['close'][i]) > self.ATR_times * data_price['atr'][i]):
                self.set_position(data_price, i, "close", '{}回落平仓'.format(max(data_price.high[i - 5:i])))

            # 情形六：买入后三个bar还不赚钱立马平仓  计步器
            elif (data_price['position'][i - 1] == 1) and ((i - self.entry_index) > 3) and (data_price['close'][i] / self.price_in < 0.985):
                self.set_position(data_price, i, "close", 'N周期未盈利平仓')

            # 情形七：盘中击穿重要支撑立马平仓
            # 作业4：把CDP指标写入到策略中，加入一个止损条件：当日价格如果击穿CDP的AL，也就是下方支撑，立即止损。
            elif (data_price['position'][i - 1] == 1) and (data_price['close'][i] < data_price['al'][i]):
                self.set_position(data_price, i, "close", '击穿CDP支撑{}止损平仓'.format(round(data_price['al'][i], 4)))

            # 情形八：逆势买入获得优势成本
            elif (data_price['position'][i - 1] == 0) and (data_price['kdj_k_ma5'][i] > 80) and (data_price['kdj_k'][i - 1] > 80) and (data_price['kdj_k'][i] < 80):
                self.set_position(data_price, i, "buy", '80上方回落买入')

            # 其他情形：保持之前的仓位不变
            else:
                data_price['position'][i] = data_price['position'][i - 1]

        p1 = pd.DataFrame(self.Buy, columns=['买入日期', '买入价格', '备注'])
        p2 = pd.DataFrame(self.Sell, columns=['卖出日期', '卖出价格', '备注'])
        transactions = pd.concat([p1, p2], axis=1)  # 交易记录

        data_price = data_price.iloc[window_long:, :]
        data_price['ret'] = data_price.close.pct_change().fillna(0)
        data_price['nav'] = (1 + data_price.ret * data_price.position.shift(1)).cumprod()  # .shift(1)下移一位
        data_price['benchmark'] = data_price.close / data_price.close[0]

        # 2.3返回交易记录和全过程数据
        return transactions, data_price

    def show_performance(self, transactions, strategy):
        # 3.1策略评价指标
        # 年化收益率
        rety = strategy.nav[strategy.shape[0] - 1] ** (self.N / strategy.shape[0]) - 1

        # 夏普比
        Sharpe = (strategy.ret * strategy.position.shift()).mean() / (strategy.ret * strategy.position.shift()).std() * np.sqrt(self.N)  # .shift(1)下移一位

        # 胜率
        VictoryRatio = ((transactions['卖出价格'] - transactions['买入价格']) > 0).mean()

        # 最大回撤率
        DD = 1 - strategy.nav / strategy.nav.cummax()
        MDD = max(DD)

        # 单次最大亏损
        maxloss = min(transactions['卖出价格'] / transactions['买入价格'] - 1)  # 单笔收益的最小值

        # 月均交易次数
        trade_count = strategy.flag.abs().sum() / strategy.shape[0] * 20 * (self.N / 250)  # 改成不同周期适用的月均交易次数

        print('------------------------------')
        print('Sharpe ratio:', round(Sharpe, 2))
        print('Annual Return:{}%'.format(round(rety * 100, 2)))
        print('Winning Rate：{}%'.format(round(VictoryRatio * 100, 2)))
        print('Maximun Drawdown：{}%'.format(round(MDD * 100, 2)))
        print('Max Single loss:{}%'.format(round(-maxloss * 100, 2)))
        print('Trading per Month：{}(Long & Short total)'.format(round(trade_count, 2)))
        print('Powered By Xavier on:{}'.format(datetime.datetime.now()))
        print('------------------------------')

        result = {'Sharpe': Sharpe,
                  'Annual_Return': rety,
                  'Winning_Rate': VictoryRatio,
                  'MDD': MDD,
                  'Max_Loss_Single_Time': -maxloss,
                  'Trading_Num': round(strategy.flag.abs().sum() / strategy.shape[0], 1)
                  }

        result = pd.DataFrame.from_dict(result, orient='index').T
        print(result)

        ##3.2策略逐年表现
        nav_peryear = strategy.nav.groupby(strategy.year).last() / strategy.nav.groupby(strategy.year).first() - 1
        benchmark_peryear = strategy.benchmark.groupby(strategy.year).last() / strategy.benchmark.groupby(strategy.year).first() - 1

        excess_ret = nav_peryear - benchmark_peryear
        result_peryear = pd.concat([nav_peryear, benchmark_peryear, excess_ret], axis=1)
        result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
        result_peryear = result_peryear.T
        print('------------------------------')
        print(result_peryear)
        print('------------------------------')

        # 3.3策略净值可视化---半成品，自适应选用的策略名称
        self.plot_nav(strategy, 'Dual_Moving_Average')

        return result, result_peryear

    # 半成品，自适应选用的策略指标线
    def plot_Kline(self, data_price, indicator):
        kline = Kline(init_opts=opts.InitOpts(width='1200px', height='600px', theme=ThemeType.DARK))  # 设置K线图的长和宽
        kline.add_xaxis(data_price.index.tolist())  # 将index也就是时间轴设置为X轴
        y = list(data_price.loc[:, ['open', 'high', 'low', 'close']].round(4).values)  # 设置为list，一共有data_price.shape[0]个，等待传入Kbar
        y = [i.tolist() for i in y]  # 里面的单个数组也必须转换成list
        kline.add_yaxis('Kline', y)
        # kline.extend_axis(yaxis=opts.AxisOpts( axislabel_opts=opts.LabelOpts(formatter="{value}") ))
        kline.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
        kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}")),
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            title_opts=opts.TitleOpts(title="{}_Kline_and_{}".format(self.symbol, indicator), pos_left='45%'),  # 题目位置
            legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        line = Line()
        line.add_xaxis(data_price.index.tolist())
        ##########################################################
        line.add_yaxis('MA_short', data_price.sma.round(2).tolist(), is_smooth=True)
        line.add_yaxis('MA_median', data_price.mma.round(2).tolist(), is_smooth=True)
        line.add_yaxis('MA_long', data_price.lma.round(2).tolist(), is_smooth=True)
        ###########################################################
        line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
        line.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )
        kline.overlap(line)
        kline.render(self.file_path + '{}_Kline_and_{}.html'.format(self.symbol, indicator))

    def plot_nav(self, strategy, indicator):
        line1 = Line(init_opts=opts.InitOpts(width='1200px', height='600px', theme=ThemeType.DARK))
        line1.add_xaxis(strategy.index.tolist())
        line1.add_yaxis('策略净值', strategy.nav.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line1.add_yaxis('基准净值', strategy.benchmark.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line1.extend_axis(yaxis=opts.AxisOpts(min_=0.8, axislabel_opts=opts.LabelOpts(formatter="{value}")))
        line1.set_series_opts(label_opts=opts.LabelOpts(is_show=True))  # 是否显示数据标签
        line1.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)),
            yaxis_opts=opts.AxisOpts(min_=0.75, axislabel_opts=opts.LabelOpts(formatter="{value}")),
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            title_opts=opts.TitleOpts(title="{}_Stratergy".format(indicator), pos_left='45%'),  # 题目位置 Dual_Moving_Average
            legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        line2 = Line()
        line2.add_xaxis(strategy.index.tolist())
        line2.add_yaxis('净值之比', (strategy.nav / strategy.benchmark).round(2).tolist(), yaxis_index=1, is_smooth=True)
        line2.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        line1.overlap(line2)
        line1.render(self.file_path + "{}_Stratergy.html".format(indicator))


if __name__ == "__main__":
    path = '/Users/aiailyu/PycharmProjects/第3节课/'
    name = '1_510050_d'
    my_strategy = Strategy(path, name, 20, 2, 'day', True) # 实例化一个类(class):Strategy
    trans, data = my_strategy.Dual_Moving_Average() # 调用Strategy类中的方法：Dual_Moving_Average()
    print('trading record：\n', trans)
    print('show the result：\n', data)
    #my_strategy.show_performance(trans, data)
