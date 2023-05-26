import numpy as np
import pandas as pd
import os
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
文件：Multi_backtesting包含工具函数和类
    strategies包含十个策略类和Backtesting执行函数

class BasicMethods: 策略的公共属性、通用函数，每个策略类继承于此类
class 策略名称(BaseMethods): 继承BaseMethods类，再增加自己的run()函数
pre_process: 某品种的数据处理程序，读文件在此处
Backtesting: 执行一个品种的所有策略
'''


class BasicMethods:
    def __init__(self, data_price, file_name, ATR_Length, ATR_times):
        '''
        :param file_name: 品种文件名。用实时数据接口的品种代码命名文件，与实时数据获取共用名称
        :param ATR_Length: ATR的计算周期
        :param ATR_times: ATR止损的乘数(倍数)
        '''
        self.symbol = file_name
        self.ATR_Length = ATR_Length
        self.ATR_times = ATR_times
        self.Buy = []  # 保存开多记录
        self.Buy_close = []  # 保存平多记录
        self.Sell = []  # 保存开空记录
        self.Sell_close = []  # 保存平空记录
        self.price_in = 1  # 初始买入价设置为1
        self.price_out = 1  # 初始卖出价设置为1
        self.buy_entry_index = 0
        self.sell_entry_index = 0
        self.data_price = data_price

    def set_position(self, data_price, i, direction, remarks):
        '''
        开多、平多、开空、平空
        :param i: 第i根K线
        :param remarks: 交易备注
        :param direction: 交易方向，"buy", "buy_close", "sell", "sell_close"
        :return:
        '''
        if direction == "buy":
            data_price['flag'][i] = 1  # 开多
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年月日
            self.price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            self.buy_entry_index = i  # 用全局变量记录这根序号
            self.Buy.append([date_in, self.price_in, remarks])  # 把买入记录保存到Buy列表里
            print(date_in, '========={}--'.format(remarks), self.price_in)
        elif direction == "buy_close":
            data_price['flag'][i] = -1  # 平多
            data_price['position'][i] = 0
            date_out = data_price.index[i]
            price_out = data_price['close'][i]
            self.Buy_close.append([date_out, price_out, remarks])
            print(date_out, '========={}--'.format(remarks), price_out)
        elif direction == "sell":
            data_price['flag'][i] = -1  # 开空
            data_price['position'][i] = -1  # 仓位记录为-1，表示有1手仓位
            date_in = data_price.index[i]  # 记录卖出的时间 年月日
            self.price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            self.sell_entry_index = i  # 用全局变量记录这根序号
            self.Sell.append([date_in, self.price_out, remarks])  # 把买入记录保存到Buy列表里
            print(date_in, '========={}--'.format(remarks), self.price_out)
        elif direction == "sell_close":
            data_price['flag'][i] = 1  # 平空
            data_price['position'][i] = 0
            date_out = data_price.index[i]
            price_out = data_price['close'][i]
            self.Sell_close.append([date_out, price_out, remarks])
            print(date_out, '========={}--'.format(remarks), price_out)
        else:
            raise ValueError("交易方向只能为buy, buy_close, sell, sell_close")

    def get_results(self, data_price, start_id):
        p1 = pd.DataFrame(self.Buy, columns=['买入日期', '买入价格', '备注'])
        p2 = pd.DataFrame(self.Buy_close, columns=['卖出日期', '卖出价格', '备注'])
        transactions_buy = pd.concat([p1, p2], axis=1)  # 做多交易记录
        p3 = pd.DataFrame(self.Sell, columns=['卖出日期', '卖出价格', '备注'])
        p4 = pd.DataFrame(self.Sell_close, columns=['买入日期', '买入价格', '备注'])
        transactions_sell = pd.concat([p3, p4], axis=1)  # 做空交易记录

        data_price = data_price.iloc[start_id:, :]
        data_price['ret'] = data_price.close.pct_change().fillna(0)
        data_price['nav'] = (1 + data_price.ret * data_price.position.shift(1)).cumprod()  # .shift(1)下移一位
        data_price['benchmark'] = data_price.close / data_price.close[0]
        return transactions_buy, transactions_sell, data_price


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

    # 顺便算出来用在R-breaker策略上
    new_df["s_setup"] = cdp + ch - cl
    new_df["b_setup"] = cdp - ch + cl
    return new_df


def pre_process(file_path, file_name, period, ATR_Length, market_code=0, real_time_stock=False):
    Dic = {'5min': (0, 12000), '15min': (1, 4000), '30min': (2, 2000), 'h': (3, 1000), 'day': (4, 250), '1min': (7, 60000)}

    def get_raw_data(data_price):
        '''
        用来get通达信最新数据并更新本地csv和pkl文件
        遍历ip_addresses连接通达信，若一个都连不上就打印提示。
        '''

        api = TdxHq_API()
        ip_addresses = {'招商证券深圳行情': '119.147.212.81',
                        '华泰证券(南京电信)': '221.231.141.60',
                        '华泰证券(上海电信)': '101.227.73.20',
                        '华泰证券(上海电信二)': '101.227.77.254',
                        '华泰证券(深圳电信)': '14.215.128.18',
                        '华泰证券(武汉电信)': '59.173.18.140',
                        '华泰证券(天津联通)': '60.28.23.80',
                        '华泰证券(沈阳联通)': '218.60.29.136',
                        '华泰证券(南京联通)': '122.192.35.44'}
        for key, val in ip_addresses.items():
            if api.connect(val, 7709):  # 注意这里的IP地址和数据接口
                print("Connected to tdx using IP address {}: {}".format(key, val))
                current_data = api.to_df(api.get_security_bars(Dic[period][0], market_code, file_name, 1, 800))  # 0:深圳，1:上海
                api.disconnect()  # 调用完以后一定要关闭接口
    
                # 连上了通达信，合并数据并排序去重
                current_data = current_data[['datetime', 'open', 'high', 'low', 'close', 'vol']]
                current_data.rename(columns={'datetime': 'timestamp', 'vol': 'volume'}, inplace=True)
                data_price = pd.concat([data_price, current_data], axis=0)
                data_price = data_price.sort_values(by='timestamp', ascending=True)
                data_price = data_price.drop_duplicates('timestamp')
                # 更新csv和pkl文件
                data_price.to_csv(file_path + file_name + '.csv', index=False)
                data_price.to_pickle(file_path + file_name + '.pkl')
                break
        else:
            input("Unable to connect to tdx using any available IP address\n按回车键继续...")  # 警告现在连不上最新数据，按照历史数据回测
        return data_price

    def data_processing(data_price):
        if period == 'day':
            data_price['timestamp'] = pd.to_datetime(data_price['timestamp']).dt.date
    
        data_price['timestamp'] = pd.to_datetime(data_price['timestamp'])
        data_price = data_price.set_index('timestamp')  # 时间设为索引
        data_price['year'] = data_price.index.year  # 增加一列年份，用于策略评价
    
        # 增加列。
        data_price['position'] = 0.0  # 记录仓位
        data_price['flag'] = 0.0  # 记录交易信号。  反手时记成了两次交易，可以改
        data_price['atr'] = ta.ATR(data_price['high'], data_price['low'], data_price['close'], ATR_Length)  # 计算self.ATR_Length周期的ATR
        data_price['al'] = CDP(data_price)['al']  # CDP的支撑位
        return data_price

    data = pd.read_csv(file_path + file_name + '.csv', usecols=['trade_date', 'open', 'high', 'low', 'close', 'volume'])
    data.rename(columns={'trade_date': 'timestamp'}, inplace=True)
    # 如果要连接接口更新本地文件
    if real_time_stock:
        data = get_raw_data(data)
    data_price = data_processing(data)
    N = Dic[period][1]
    return data_price, N


def show_performance(strategy_name, transactions_buy, transactions_sell, strategy, N, file_path):
    # 3.1策略评价指标
    # 年化收益率
    rety = strategy.nav[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1

    # 夏普比
    Sharpe = (strategy.ret * strategy.position.shift()).mean() / (strategy.ret * strategy.position.shift()).std() * np.sqrt(N)  # .shift(1)下移一位

    # 胜率
    VictoryRatio = (len(transactions_buy.query("(卖出价格 - 买入价格) > 0")) + len(transactions_sell.query("(买入价格 - 卖出价格) > 0"))) / (len(transactions_buy) + len(transactions_sell))


    # 最大回撤率
    DD = 1 - strategy.nav / strategy.nav.cummax()
    MDD = max(DD)

    # 单次最大亏损
    maxloss = min(min(transactions_buy['卖出价格'] / transactions_buy['买入价格'] - 1), min(1 - transactions_sell['买入价格'] / transactions_sell['卖出价格']))  # 单笔收益的最小值

    # 月均交易次数
    trade_count = strategy.flag.abs().sum() / strategy.shape[0] * 20 * (N / 250)  # 改成不同周期适用的月均交易次数

    print('------------------------------\n策略名称: {}\n'.format(strategy_name))
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

    # 3.2策略逐年表现
    nav_peryear = strategy.nav.groupby(strategy.year).last() / strategy.nav.groupby(strategy.year).first() - 1
    benchmark_peryear = strategy.benchmark.groupby(strategy.year).last() / strategy.benchmark.groupby(strategy.year).first() - 1

    excess_ret = nav_peryear - benchmark_peryear
    result_peryear = pd.concat([nav_peryear, benchmark_peryear, excess_ret], axis=1)
    result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
    result_peryear = result_peryear.T
    print('------------------------------')
    print(result_peryear)
    print('------------------------------')

    # 3.3策略净值可视化
    plot_nav(strategy, strategy_name, file_path)
    return Sharpe


def plot_nav(strategy, indicator, file_path):
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
    line1.render(file_path + "{}_Stratergy.html".format(indicator))


# 画K线和策略指标线。暂时不用
def plot_Kline(data_price, symbol, indicator, file_path):
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
        title_opts=opts.TitleOpts(title="{}_Kline_and_{}".format(symbol, indicator), pos_left='45%'),  # 题目位置
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
    kline.render(file_path + '{}_Kline_and_{}.html'.format(symbol, indicator))
