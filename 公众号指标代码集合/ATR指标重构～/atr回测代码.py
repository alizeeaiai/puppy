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


# 数据处理函数
def data_process():
    file_path = './trends_added.csv'
    df = pd.read_csv(file_path)
    # 把timestamp变成时间序列
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['year'] = df.index.year
    # df = df.drop(['support', 'H-L', 'SMA', 'HiLo', 'Href', 'diff1', 'diff2', 'ATRmod', 'loss', 'resistance', 'Lref', 'open', 'volume'], axis=1)
    return df


def my_strategy(data_price):
    
    Buy = []  # 保存买入记录
    Sell = []  # 保存卖出记录
    price_in = 1  # 初始买入价设置为1
    data_price['position'] = 0
    data_price['flag'] = 0
    for i in range(1, data_price.shape[0]):
        # 买入情形:当最高价站上trends则买入
        if (data_price['position'][i - 1] == 0) & (data_price['high'][i] > data_price['trends'][i]):
            data_price['flag'][i] = 1  # 记录做多还是做空，这里1是做多
            data_price['position'][i] = 1 # 仓位记录为1，表示有1手仓位
            date_in = data_price.index[i]  # 记录买入的时间 年-月-日
            price_in = data_price['close'][i]  # 记录买入的价格，这里是以收盘价买入
            entry_index = i
            print(data_price.index[i], '价格站上均线买入', price_in)
            Buy.append([date_in, price_in, '价格站上均线则买入'])  # 把买入记录保存到Buy列表里
            # 上述也都可以使用data_price.at[i, 'position']的用法，为了确保没有错误，暂且这么使用

        # 平仓规则，当最「最高价」跌破atr时，才发出卖出信号
        elif (data_price['position'][i-1] == 1) & (data_price['high'][i] < data_price['trends'][i]):
            print(data_price.index[i], '最高价跌穿ATR卖出')
            data_price['flag'][i] = -1  # 记录做多还是做空，这里-1是卖出
            data_price['position'][i] = 0  # 仓位记录为0，表示没有仓位了
            date_out = data_price.index[i]  # 记录卖出的时间 年-月-日
            price_out = data_price['close'][i]  # 记录卖出的价格，这里是以收盘价卖出
            Sell.append([date_out, price_out, '最高价跌穿atr卖出'])  # 把卖出记录保存到Sell列表里

        # 其他情形：保持之前的仓位不变
        else:
            data_price['position'][i] = data_price['position'][i - 1]
            print(data_price.index[i], '============没有买卖，继续持仓')

    p1 = pd.DataFrame(Buy, columns=['买入日期', '买入价格', '备注'])
    p2 = pd.DataFrame(Sell, columns=['卖出日期', '卖出价格', '备注'])
    transactions = pd.concat([p1, p2], axis=1)  # 交易记录

    data_price['position'] = data_price['position'].shift(1).fillna(0)
    data_price['ret'] = data_price.close.pct_change(1).fillna(0)
    data_price['nav'] = (1 + data_price.ret * data_price.position).cumprod()
    data_price['benchmark'] = data_price.close / data_price.close[0]
    return transactions, data_price

def show_performance(transactions, strategy):
    ##3.1策略评价指标
    # 年化收益率
    N = 250
    rety = strategy.nav[strategy.shape[0] - 1] ** (N / strategy.shape[0]) - 1

    # 夏普比
    Sharp = (strategy.ret * strategy.position).mean() / (strategy.ret * strategy.position).std() * np.sqrt(N)

    # 胜率
    VictoryRatio = ((transactions['卖出价格'] - transactions['买入价格']) > 0).mean()

    # 最大回撤率
    DD = 1 - strategy.nav / strategy.nav.cummax()
    MDD = max(DD)

    # 单次最大亏损
    maxloss = min(transactions['卖出价格'] / transactions['买入价格'] - 1)

    # 月均交易次数
    trade_count = strategy.flag.abs().sum() / strategy.shape[0] * 20

    print('------------------------------')
    print('Sharpe ratio:', round(Sharp, 2))
    print('Annual Return:{}%'.format(round(rety * 100, 2)))
    print('Winning Rate：{}%'.format(round(VictoryRatio * 100, 2)))
    print('Maximun Drawdown：{}%'.format(round(MDD * 100, 2)))
    print('Max Single loss:{}%'.format(round(-maxloss * 100, 2)))
    print('Trading per Month：{}(Long & Short total)'.format(round(trade_count, 2)))
    print('Powered By Puppy on:{}'.format(datetime.datetime.now()))
    print('------------------------------')

    result = {'Sharp': Sharp,
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
    benchmark_peryear = strategy.benchmark.groupby(strategy.year).last() / strategy.benchmark.groupby(
        strategy.year).first() - 1

    excess_ret = nav_peryear - benchmark_peryear
    result_peryear = pd.concat([nav_peryear, benchmark_peryear, excess_ret], axis=1)
    result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
    result_peryear = result_peryear.T
    print('------------------------------')
    print(result_peryear)
    print('------------------------------')

    ##3.3策略净值可视化
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
        title_opts=opts.TitleOpts(title="Dual_Moving_Average_Stratergy", pos_left='45%'),  # 题目位置
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
    line1.render('./new_atr_version.html')

    return result, result_peryear


if __name__ == '__main__':
    data = data_process()
    trans, strategy = my_strategy(data)
    show_performance(trans, strategy)
