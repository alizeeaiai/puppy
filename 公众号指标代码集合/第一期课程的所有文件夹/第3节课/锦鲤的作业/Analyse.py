import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import time
import datetime

class Analyse:

    # 策略回测分析
    # df_price : 回测过程数据
    # order_list : 回测交易记录数据
    # merge_order : 交易记录合并数据
    # N : 每年交易天数
    def __init__(self, df_price, order_list, merge_order, rf=0.04, N=252):
        self.df_price = df_price.copy()
        self.order_list = order_list.copy()
        self.merge_order = merge_order.copy()
        self.risk_free_rate = rf
        self.period_N = N

        self.df_price['year'] = self.df_price.index.year

        # 删除空行数据
        self.merge_order.dropna(axis=0, how='any', inplace=True)

    def show_performance(self):

        # 年化收益率
        annualized_return = self.df_price.net_value[self.df_price.shape[0] - 1] ** (self.period_N / self.df_price.shape[0]) - 1

        # 夏普比率
        Sharp = (annualized_return - self.risk_free_rate) / ((self.df_price.change_pct * self.df_price.position).std() * np.sqrt(self.period_N))

        # 胜率
        profit_list = ((self.merge_order['平仓价格'] - self.merge_order['开仓价格']) / self.merge_order['开仓价格']) * self.merge_order['开仓仓位']
        VictoryRatio = (len([i for i in profit_list if i > 0])) / self.merge_order.shape[0]

        # 最大回撤率
        DD = 1 - self.df_price.net_value / self.df_price.net_value.cummax()
        MDD = max(DD)

        # 单次最大亏损
        MaxLoss = min(profit_list)

        # 单次最大盈利
        MaxProfit = max(profit_list)

        # 月均交易次数
        TradeCountMonthly = (self.df_price.trans_flag.abs().sum() * 20)/ self.df_price.shape[0]

        print('------------------------------')
        print('Sharpe ratio:', round(Sharp, 2))
        print('Annual Return:{}%'.format(round(annualized_return * 100, 2)))
        print('Winning Rate：{}%'.format(round(VictoryRatio * 100, 2)))
        print('Maximun Drawdown：{}%'.format(round(MDD * 100, 2)))
        print('Max Single loss:{}%'.format(round(-MaxLoss * 100, 2)))
        print('Max Single profit:{}%'.format(round(MaxProfit * 100, 2)))
        print('Trading per Month：{}(Long & Short total)'.format(round(TradeCountMonthly, 2)))
        print('Powered By L on:{}'.format(datetime.datetime.now()))
        print('------------------------------')

        # 策略逐年表现
        nav_yearly = self.df_price.net_value.groupby(self.df_price.year).last() / self.df_price.net_value.groupby(self.df_price.year).first() - 1
        benchmark_yearly = self.df_price.benchmark.groupby(self.df_price.year).last() / self.df_price.benchmark.groupby(self.df_price.year).first() - 1

        excess_ret = nav_yearly - benchmark_yearly
        result_peryear = pd.concat([nav_yearly, benchmark_yearly, excess_ret], axis=1)
        result_peryear.columns = ['strategy_ret', 'bench_ret', 'excess_ret']
        # result_peryear = result_peryear.T
        print('------------------------------')
        print(result_peryear)
        print('------------------------------')

    # 策略净值可视化
    def plot(self, file_path, file_name):
        line1 = Line(init_opts=opts.InitOpts(width='2000px', height='800px', theme=ThemeType.DARK))
        line1.add_xaxis(self.df_price.index.tolist())
        line1.add_yaxis('策略净值', self.df_price.net_value.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line1.add_yaxis('基准净值', self.df_price.benchmark.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line1.extend_axis(yaxis=opts.AxisOpts(min_=0.8, axislabel_opts=opts.LabelOpts(formatter="{value}")))
        line1.set_series_opts(label_opts=opts.LabelOpts(is_show=True))  # 是否显示数据标签
        line1.set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)),
            yaxis_opts=opts.AxisOpts(min_=0.00, axislabel_opts=opts.LabelOpts(formatter="{value}")),
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            title_opts=opts.TitleOpts(title="Dual_Moving_Average_Stratergy", pos_left='45%'),  # 题目位置
            legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        line2 = Line()
        line2.add_xaxis(self.df_price.index.tolist())
        line2.add_yaxis('净值之比', (self.df_price.net_value / self.df_price.benchmark).round(2).tolist(), yaxis_index=1, is_smooth=True)
        line2.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        line1.overlap(line2)
        line1.render(file_path + file_name)
