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


start_time = time.time()
file_path = './'
df_price = pd.read_excel(file_path + '1_510050_d.xlsx')
df_price.rename(columns={'etime': 'timestamp'}, inplace=True)
# print(df_price)

api = TdxHq_API()
if api.connect('119.147.212.81', 7709):  # 注意这里的IP地址和数据接口
    # 注意这里，第一个1表示是15分钟的数据，其中0为5分钟K线 1 15分钟K线 2 30分钟K线 3 1小时K线 4 日K线
    current_data = api.to_df(api.get_security_bars(4, 1, '510050', 1, 200))
    api.disconnect()  # 调用完以后一定要关闭接口

# 4、提取current_data的数据
current_data = current_data[['datetime',
                             'open', 'high', 'low', 'close', 'vol']]
current_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# add01 pd.to_datetime() 是 Pandas 库中的一个函数，用于将一个字符串、日期、时间戳等转换为 Pandas 中的日期时间类型（DatetimeIndex），以方便进行时间序列数据的处理和分析。
# 虽然转换前可能已是 Pandas 中的日期时间类型（DatetimeIndex），但转换一下总是好的。
current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
df_price = pd.concat([df_price, current_data], axis=0)  # 合并数据
del df_price['amount']
# add02，将timestamp统一为日期单位，省略掉时分秒
df_price['timestamp'] = df_price['timestamp'].dt.date
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
# 注意这一步是非常必要的，要以timestamp作为排序基准
df_price = df_price.sort_values(by='timestamp', ascending=True)
df_price = df_price.drop_duplicates(
    'timestamp').reset_index()  # 注意这一步非常重要，以timestamp为基准进行去重处理
del df_price['index']
df_price = df_price.set_index('timestamp')
df_price['year'] = df_price.index.year

def MA_Strategy(data_price, window_short=5, window_median=10, window_long=20, loss_ratio=0.20):
    # df_price:价格数据；
    # window_short：短均线周期，默认为5；
    # window_long：长均线周期，默认为10；
    # lossratio：止损率,默认为1%，即开仓后下跌超过1%止损。
    # 2.0 get基础数据并计算
    data_price = data_price.copy()
    # add03把index改为年月日的形式，可能跟02一样的效果
    data_price.index = data_price.index.strftime('%Y%m%d')
    # 使用talib算法计算技术指标
    data_price['sma'] = ta.MA(
        data_price['close'], timeperiod=window_short, matype=0)
    data_price['mma'] = ta.MA(
        data_price['close'], timeperiod=window_median, matype=0)
    data_price['kdj_k'], data_price['kdj_d'] = ta.STOCH(
        data_price['high'], data_price['low'], data_price['close'], fastk_period=9, slowk_period=5, slowk_matype=1, slowd_period=5, slowd_matype=1)
    data_price['kdj_k_ma5'] = ta.MA(
        data_price['kdj_k'], timeperiod=window_short, matype=0)
    data_price['lma'] = ta.MA(
        data_price['close'], timeperiod=window_long, matype=0)
    data_price['slope'] = ta.LINEARREG_SLOPE(data_price["lma"], 5)

    data_price['position'] = 0.0  # 记录仓位
    data_price['flag'] = 0.0  # 记录买卖
    data_price = data_price.replace([np.inf, -np.inf, np.nan], 0.0)
    # print(data_price['sma'], '=========sma')
    # print(data_price['lma'], '=========lma')
    # print(data_price['mma'], '=========mma')
    print(data_price.tail(40))

    # 2.1绘制K线和均线

    kline = Kline(init_opts=opts.InitOpts(width='1200px',
                  height='600px', theme=ThemeType.DARK))  # 设置K线图的长和宽
    kline.add_xaxis(data_price.index.tolist())  # 将index也就是时间轴设置为X轴
    y = list(data_price.loc[:, ['open', 'high', 'low', 'close']].round(
        4).values)  # 设置为list，一共有data_price.shape[0]个，等待传入Kbar
    y = [i.tolist() for i in y]  # 里面的单个数组也必须转换成list
    kline.add_yaxis('Kline', y)
    #kline.extend_axis(yaxis=opts.AxisOpts( axislabel_opts=opts.LabelOpts(formatter="{value}") ))
    kline.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
    kline.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)),
        yaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value}")),
        datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
        title_opts=opts.TitleOpts(
            title="510050_50ETF_Kline_and_MA", pos_left='45%'),  # 题目位置
        legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross")  # 添加趋势线
    )

    line = Line()
    line.add_xaxis(data_price.index.tolist())
    line.add_yaxis('MA_short', data_price.sma.round(
        2).tolist(), is_smooth=True)
    line.add_yaxis('MA_median', data_price.mma.round(
        2).tolist(), is_smooth=True)
    line.add_yaxis('MA_long', data_price.lma.round(2).tolist(), is_smooth=True)
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
    line.set_global_opts(
        datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
        legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross")  # 添加趋势线
    )
    kline.overlap(line)
    kline.render(file_path + '510050_50ETF_Kline_and_MA.html')

    # 2.2均线策略的交易记录
    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0

    for i in range(max(1, window_long), data_price.shape[0]):
        data_price = data_price.copy()

        buy_cond_1 = (data_price['sma'][i - 1] < data_price['lma'][i - 1]
                      ) and (data_price['sma'][i] > data_price['lma'][i])  # 金叉
        # add04，前面部分总结起来就是：当长期均线斜率向上，同时当前bar发生短期均线死叉中期均线。再加上最后一个条件，sma大于lma。
        # 两个逻辑定义现在为多头行情，第一是长期均线斜率向上。第二是短期均线在长期均线上方。多了一个意义是没有调整到长期均价下方，说明不光是多头，还是比较强筋的多头。这时候如果sma死叉了mma，就认为是调整到位。
        # 如果看到价格突破均线或者金叉均线，或者sma金叉lma才买。这时候成功率只有30多趴。这么做不好并不是因为胜率，而在于赔率。因为这时买行情已经走了一大截。
        # buy_cond_2 = (data_price['lma'][i] > data_price['lma'][i-1]) and (data_price['sma'][i - 1] > data_price['mma'][i - 1]) and (
        # 作业二，使用此方法data_price['slope'][i] > 0 替换原方法后，回报从466倍变成了510倍。
        buy_cond_2 = (data_price['slope'][i] > 0) and (data_price['sma'][i - 1] > data_price['mma'][i - 1]) and (
            data_price['sma'][i] < data_price['mma'][i]) and (data_price['sma'][i] > data_price['lma'][i])
        # 用slope来表示5个周期的20日均线向上，talib，唯快不破，
        # if buy_cond_2:
        #     print(data_price.index[i], '=====condtion_2=========')

        # print(data_price.index[i], '============', max(data_price.high[i-5:i]))

        # 情形一：当前无仓位且短均线上穿长均线(金叉)，则买入股票
        if (data_price['position'][i-1] == 0) and (data_price['sma'][i - 1] < data_price['lma'][i - 1]) and (
                data_price['sma'][i] > data_price['lma'][i]):

            data_price['flag'][i] = 1  # 记录做多还是做空，这里1是做多
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            entry_index = i
            print(data_price.index[i], '=========金叉买入@--', price_in)
            Buy.append([date_in, price_in, '金叉买入'])  # 把买入记录保存到Buy列表里
            # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

        elif (data_price['position'][i-1] == 0) and buy_cond_2:
            print(data_price.index[i], '=========多头死叉买入')
            data_price['flag'][i] = 1  # 记录做多还是做空，这里1是做多
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            entry_index = i
            Buy.append([date_in, price_in, '多头死叉买入'])  # 把买入记录保存到Buy列表里
            # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

        # 情形二：当前持仓且下跌超过止损率，则平仓止损
        elif (data_price['position'][i-1] == 1) and ((1.0 - data_price['close'][i] / price_in) > loss_ratio):
            print(data_price.index[i], '=========止损平仓')
            data_price['flag'][i] = -1  # 记录做多还是做空，这里-1是做多
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '止损平仓'])  # 把卖出记录保存到Sell列表里
            # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

        # 情形三：当前持仓且短均线下穿长均线(死叉)，则卖出股票
        elif (data_price['position'][i-1] == 1) & (data_price['sma'][i - 1] > data_price['lma'][i - 1]) & (
                data_price['sma'][i] < data_price['lma'][i]):
            print(data_price.index[i], '=========死叉卖出')
            data_price['flag'][i] = -1  # 记录做多还是做空，这里-1是做多
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里

        # 情形四：当前持仓且短均线下穿长均线(死叉)，则卖出股票
        elif (data_price['position'][i-1] == 1) & (data_price['sma'][i - 1] > data_price['lma'][i - 1]) & (
                data_price['sma'][i] < data_price['lma'][i]):
            # print(data_price.index[i], '=========死叉卖出')
            data_price['flag'][i] = -1  # 记录做多还是做空，这里-1是做多
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '死叉卖出'])  # 把卖出记录保存到Sell列表里

        # 情形五：高点回落大于阈值时平仓
        # 吊灯止盈法--吊灯的设置：固定/浮动
        # 第二个作业：开发N倍ATR作为高点回落出局条件
        elif (data_price['position'][i-1] == 1) and ((max(data_price.high[i-5:i]) / data_price['close'][i] - 1) > loss_ratio*0.8):
            print(data_price.index[i], '============', max(
                data_price.high[i-5:i]), '=========回落平仓')
            data_price['flag'][i] = -1  # 记录做多还是做空，这里-1是做多
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '回落平仓'])  # 把卖出记录保存到Sell列表里

        # 情形六：买入后三个bar还不赚钱立马平仓  计步器
        elif (data_price['position'][i-1] == 1) and (i - entry_index > 3) and (data_price['close'][i] / price_in < 0.985):
            # print(i - entry_index, ' hodling for a period')
            data_price['flag'][i] = -1  # 记录做多还是做空，这里-1是做多
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            # print(data_price.index[i], '============', '=========持仓超过3个周期还未盈利达到1%平仓@--', price_out)
            Sell.append([date_out, price_out, 'N周期未盈利平仓'])  # 把卖出记录保存到Sell列表里

        # 情形七：盘中击穿重要支撑立马平仓

        # 情形八：逆势买入获得优势成本
        elif (data_price['position'][i-1] == 0) and (data_price['kdj_k_ma5'][i] > 80) and (data_price['kdj_k'][i-1] > 80) and (data_price['kdj_k'][i] < 80):
            data_price['flag'][i] = 1  # 记录做多还是做空，这里1是做多
            data_price['position'][i] = 1  # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            entry_index = i
            print(data_price.index[i], '=========80上方回落买入@--', price_in)
            Buy.append([date_in, price_in, '80上方回落买入'])  # 把买入记录保存到Buy列表里

        # 其他情形：保持之前的仓位不变
        else:
            data_price['position'][i] = data_price['position'][i - 1]
            # print(data_price.index[i], '============没有买卖，继续持仓')

        # print(data_price.index[i], '======================', data_price['position'][i])
    # print(data_price.tail(60))
    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # 交易记录

    data_price = data_price.iloc[window_long:, :]
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    data_price['nav'] = (1 + data_price.ret*data_price.position).cumprod()
    data_price['benchmark'] = data_price.close/data_price.close[0]

    # 2.3返回交易记录和全过程数据
    return transactions, data_price


def show_performance(transactions, strategy):
    # 3.1策略评价指标
    # 年化收益率
    N = 250
    rety = strategy.nav[strategy.shape[0] - 1]**(N/strategy.shape[0]) - 1

    # 夏普比
    Sharp = (strategy.ret*strategy.position).mean() / \
        (strategy.ret*strategy.position).std()*np.sqrt(N)

    # 胜率
    VictoryRatio = ((transactions['卖出价格'] - transactions['买入价格']) > 0).mean()

    # 最大回撤率
    DD = 1 - strategy.nav/strategy.nav.cummax()
    MDD = max(DD)

    # 单次最大亏损
    maxloss = min(transactions['卖出价格']/transactions['买入价格'] - 1)

    # 月均交易次数
    trade_count = strategy.flag.abs().sum()/strategy.shape[0]*20

    print('------------------------------')
    print('Sharpe ratio:', round(Sharp, 2))
    print('Annual Return:{}%'.format(round(rety*100, 2)))
    print('Winning Rate：{}%'.format(round(VictoryRatio*100, 2)))
    print('Maximun Drawdown：{}%'.format(round(MDD*100, 2)))
    print('Max Single loss:{}%'.format(round(-maxloss*100, 2)))
    print('Trading per Month：{}(Long & Short total)'.format(round(trade_count, 2)))
    print('Powered By Xavier on:{}'.format(datetime.datetime.now()))
    print('------------------------------')

    result = {'Sharp': Sharp,
              'Annual_Return': rety,
              'Winning_Rate': VictoryRatio,
              'MDD': MDD,
              'Max_Loss_Single_Time': -maxloss,
              'Trading_Num': round(strategy.flag.abs().sum()/strategy.shape[0], 1)
              }

    result = pd.DataFrame.from_dict(result, orient='index').T
    print(result)

    # 3.2策略逐年表现
    nav_peryear = strategy.nav.groupby(strategy.year).last(
    )/strategy.nav.groupby(strategy.year).first() - 1
    benchmark_peryear = strategy.benchmark.groupby(
        strategy.year).last()/strategy.benchmark.groupby(strategy.year).first() - 1

    excess_ret = nav_peryear - benchmark_peryear
    result_peryear = pd.concat(
        [nav_peryear, benchmark_peryear, excess_ret], axis=1)
    result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
    result_peryear = result_peryear.T
    print('------------------------------')
    print(result_peryear)
    print('------------------------------')

    # 3.3策略净值可视化
    line1 = Line(init_opts=opts.InitOpts(
        width='1200px', height='600px', theme=ThemeType.DARK))
    line1.add_xaxis(strategy.index.tolist())
    line1.add_yaxis('策略净值', strategy.nav.round(
        2).to_list(), yaxis_index=0, is_smooth=True)
    line1.add_yaxis('基准净值', strategy.benchmark.round(
        2).to_list(), yaxis_index=0, is_smooth=True)
    line1.extend_axis(yaxis=opts.AxisOpts(
        min_=0.8, axislabel_opts=opts.LabelOpts(formatter="{value}")))
    line1.set_series_opts(label_opts=opts.LabelOpts(is_show=True))  # 是否显示数据标签
    line1.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)),
        yaxis_opts=opts.AxisOpts(
            min_=0.75, axislabel_opts=opts.LabelOpts(formatter="{value}")),
        datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
        title_opts=opts.TitleOpts(
            title="Dual_Moving_Average_Stratergy", pos_left='45%'),  # 题目位置
        legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross")  # 添加趋势线
    )

    line2 = Line()
    line2.add_xaxis(strategy.index.tolist())
    line2.add_yaxis('净值之比', (strategy.nav/strategy.benchmark).round(2).tolist(),
                    yaxis_index=1, is_smooth=True)
    line2.set_global_opts(
        datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
        legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross")  # 添加趋势线
    )

    line1.overlap(line2)
    line1.render(file_path + "Dual_Moving_Average_Stratergy.html")

    return result, result_peryear


trans, data = MA_Strategy(df_price, window_short=3,window_median=8, window_long=13, loss_ratio=0.05)
print('trading record：\n', trans)
print('show the result：\n', data)
# show_performance(trans, data)
#
# # 规则类策略第一步应该是测试这个品种是不是好做
# # 1、10个品种，10个技术指标，金叉死叉作为买卖条件，用来选择品种；输出最终结果，看看哪几个品种的sharpe最高
# # 2、用slope来表示5个周期的20日均线向上——（1）talib查询slope函数，dataframe里面，生成slope
# # 3、开发N倍ATR作为高点回落出局条件；
# # 4、把CDP指标写入到策略中，加入一个止损条件：当日价格如果击穿CDP的AL，也就是下方支撑，立即止损
#
# end_time = time.time()
# print('time cost:---------', end_time-start_time, '   s----------')
