import pandas as pd
import numpy as np
import tushare as ts
import warnings
import talib as ta
from datetime import datetime
from sklearn.linear_model import LinearRegression
from pyecharts.components import Table
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.options import ComponentTitleOpts
from pyecharts.globals import ThemeType
from pyecharts.components import Table
warnings.filterwarnings('ignore')

# 获取每日成交金额
df = pd.read_csv('./股票数据大集合/宁德时代每日成交金额.csv')
new_df = df[['trade_date', 'close', 'amount', 'vol', 'pct_chg']]

# 计算每日的D值，D=每日成交金额/每日成交笔数
new_df['D'] = new_df['amount'] / new_df['vol']

# 按每个月对数据分组，一个月分为30天，二月份全用28天
new_df = new_df.sort_values(by='trade_date')
new_df.reset_index(drop=True, inplace=True)
new_df['trade_date'] = pd.to_datetime(new_df['trade_date'], format='%Y%m%d')
new_df = new_df.set_index('trade_date')

dfs = []
for year_month in new_df.resample('M').groups:
    start_date = year_month.strftime('%Y%m') + '01'
    if year_month.month == 2:
        end_date = year_month.strftime('%Y%m') + '28'
    else:
        end_date = year_month.strftime('%Y%m') + '30'
    month_df = new_df.loc[start_date:end_date]
    dfs.append(month_df)

month_df_li = []
# 对每个月的D值排序
for i, month_df in enumerate(dfs[:]):
    month_df = month_df.sort_values(by='D', ascending=False)
    # 前D/2个涨跌幅相加，得到M_high; 后D/2个涨跌幅相加，得到M_low
    month_df.reset_index(inplace=True)
    M_high = month_df['pct_chg'].iloc[:10].sum()
    M_low = month_df['pct_chg'].iloc[10:].sum()
    # 插入新列fct
    month_df['fct'] = np.where(month_df.index < 10, M_high, M_low)
    # 去掉没用的列
    month_df = month_df.drop(['amount', 'vol', 'D'], axis=1)
    # 重置索引
    month_df = month_df.sort_values(by='trade_date')
    month_df.reset_index(drop=True, inplace=True)
    month_df_li.append(month_df)
concatenated_df = pd.concat(month_df_li, axis=0)
concatenated_df.reset_index(drop=True, inplace=True)
# 下面写因子ic计算
ic_periods = 80
concatenated_df['benchmark'] = concatenated_df.close / concatenated_df.close[0]
concatenated_df['ic'] = concatenated_df['fct'].rolling(ic_periods).corr(concatenated_df['pct_chg'])
concatenated_df['ic_ma'] = ta.MA(concatenated_df['ic'], timeperiod=120)
# 初始化线性回归模型，划分训练集和测试集
linear_model = LinearRegression(fit_intercept=True)
train_set_bgn_time = datetime.strptime('2019-01-01', '%Y-%m-%d')
train_set_end_time = datetime.strptime('2020-01-01', '%Y-%m-%d')
test_set_bgn_time = datetime.strptime('2020-01-02', '%Y-%m-%d')
test_set_end_time = datetime.strptime('2023-04-20', '%Y-%m-%d')
train_set = concatenated_df[(concatenated_df['trade_date'] >= train_set_bgn_time) &
                                         (concatenated_df['trade_date'] <= train_set_end_time)]
test_set = concatenated_df[(concatenated_df['trade_date'] >= test_set_bgn_time) &
                                         (concatenated_df['trade_date'] <= test_set_end_time)]
# 划分x train、y train和x test、y test
x_train = train_set['fct'].values.reshape(-1, 1)
y_train = train_set['pct_chg'].values.reshape(-1, 1)
# 模型拟合训练集
linear_model.fit(x_train, y_train)
y_train_hat = linear_model.predict(x_train)
# 测试集预测
x_test = test_set['fct'].values.reshape(-1, 1)
y_test_hat = linear_model.predict(x_test)
# 总体数据预测
x_all = concatenated_df['fct'].values.reshape(-1, 1)
all_y_hat = linear_model.predict(x_all)

train_set['y_hat'] = [i[0] for i in y_train_hat]
test_set['y_hat'] = [i[0] for i in y_test_hat]
concatenated_df['y_hat'] = [i[0] for i in all_y_hat]


def analyse(data_set, set_name):

    pos_coef = 40
    # 计算每日预测仓位
    data_set['predict_position'] = data_set['y_hat'] * pos_coef
    # 把仓位控制在-1到1，不加杠杆
    data_set['predict_position'] = data_set['predict_position'].clip(-1, 1)
    data_set['position'] = data_set['predict_position'].shift(1)
    data_set = data_set.replace([np.inf, -np.inf, np.nan], 0.0)
    date_list = [i for i in concatenated_df['trade_date'].dt.date.unique()]
    data_set['last_pos'] = data_set['position'].shift(1)
    data_set = data_set.replace([np.inf, -np.inf, np.nan], 0.0)
    data_set['inc_pos'] = data_set['position'] - data_set['last_pos']
    data_set['nav'] = (1 + (data_set.last_pos + data_set.inc_pos * (1 - 0.01)) * data_set.pct_chg).cumprod()

    # 计算每个周期的收益额以及收益率
    data_set['last_nav'] = data_set['nav'].shift(1)
    data_set = data_set.replace([np.inf, -np.inf, np.nan], 0.0)
    data_set['inc_nav'] = data_set['nav'] - data_set['last_nav']
    data_set['profit'] = data_set.nav.pct_change(1).fillna(0)

    indicators_frame = pd.DataFrame()

    # 交易日列表
    date_list = [i for i in data_set['trade_date'].dt.date.unique()]

    # 每日净值列表
    nav_daily = [i for i in data_set.groupby(['trade_date']).tail(1).nav]
    df_nav_daily = pd.DataFrame({'tdate': date_list, 'nav': nav_daily})

    # 计算每日收益率
    df_nav_daily['return_daily'] = df_nav_daily.nav.pct_change(1).fillna(0)

    # 总收益率
    total_return = df_nav_daily.nav[df_nav_daily.shape[0] - 1] / df_nav_daily.nav[0] - 1

    # 年化收益率
    annual_return = (1 + total_return) ** (250 / (len(date_list))) - 1

    # 夏普比率
    sharp_ratio = (annual_return - 0.03) / (df_nav_daily['return_daily'].std() * np.sqrt(250))

    # 最大回撤率及起始日期
    drawdown_list = 1 - df_nav_daily.nav / df_nav_daily.nav.cummax()
    maximum_drawdown = max(drawdown_list)
    mdd_end_date = date_list[np.argmax(drawdown_list)]
    mdd_start_date = date_list[np.argmax(nav_daily[: np.argmax(drawdown_list)])]

    # 卡尔玛比率
    calmar_ratio = (annual_return - 0.03) / maximum_drawdown

    # 总交易次数
    total_trading_times = len(data_set[data_set['position'] != 0])
    win_data = data_set[data_set['profit'] > 0]
    loss_data = data_set[data_set['profit'] < 0]

    # 胜率
    winning_rate = len(win_data) / total_trading_times

    # 平均每笔交易盈亏比
    gain_loss_ratio = win_data['profit'].mean() / abs(loss_data['profit'].mean())

    # 总盈亏比
    total_gain_loss_ratio = win_data['inc_nav'].sum() / abs(loss_data['inc_nav'].sum())

    # IC均值
    ic = data_set['ic'].mean()

    # IR
    ir = data_set['ic'].mean() / data_set['ic'].std()

    indicators_frame = pd.DataFrame()
    indicators_frame.loc[set_name, '总收益率'] = total_return
    indicators_frame.loc[set_name, '年化收益率'] = annual_return
    indicators_frame.loc[set_name, '夏普比率'] = sharp_ratio
    indicators_frame.loc[set_name, '卡尔玛比率'] = calmar_ratio
    indicators_frame.loc[set_name, '最大回撤率'] = maximum_drawdown
    indicators_frame.loc[set_name, '最大回撤开始日期'] = mdd_start_date
    indicators_frame.loc[set_name, '最大回撤结束日期'] = mdd_end_date
    indicators_frame.loc[set_name, '胜率'] = winning_rate
    indicators_frame.loc[set_name, '平均每笔交易盈亏比'] = gain_loss_ratio
    indicators_frame.loc[set_name, '总盈亏比'] = total_gain_loss_ratio
    indicators_frame.loc[set_name, 'IC'] = ic
    indicators_frame.loc[set_name, 'IR'] = ir

    return data_set, indicators_frame




class draw_picture:
    def __init__(self, data_set, indicators_frame):
        # 设置回测报告画布大小，不同大小的显示屏可以设置不同的参数
        self.figure_width = '1400px'
        self.figure_height = '600px'
        self.data_set = data_set
        self.indicators_frame = indicators_frame

    # 绘制回测结果统计表
    def statistical_table(self):
        indicators_frame = self.indicators_frame.reset_index()
        indicators_frame = indicators_frame.round(2)

        headers = list(indicators_frame.columns)
        rows = [list(indicators_frame.at[i]) for i in indicators_frame.index]

        table = Table()
        table.add(
            headers,
            rows,
            attributes={
                "style": "width:1400px; height:600px; text-align:center;"
            }
        )
        table.set_global_opts(
            title_opts=ComponentTitleOpts(title="回测指标统计", title_style={
                "style": "font-size: 18px; font-weight: bold; text-align:center;"
            })
        )
        return table

    # 绘制净值曲线图
    def nav_line(self, data_set, set_name):
        # 策略净值曲线
        line1 = Line(init_opts=opts.InitOpts(width=self.figure_width, height=self.figure_height, theme=ThemeType.DARK))
        line1.add_xaxis(data_set['trade_date'].tolist())
        line1.add_yaxis('策略净值', data_set.nav.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line1.set_series_opts(label_opts=opts.LabelOpts(is_show=True))  # 是否显示数据标签
        line1.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                is_scale=True,
                axislabel_opts=opts.LabelOpts(rotate=60),
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                split_number=20,
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                min_=0.00,
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(type_='inside', xaxis_index=[0, 0]),  # 内部滑动
                opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1]),
            ],
            title_opts=opts.TitleOpts(title='%s净值曲线图'%set_name, pos_left='45%'),  # 题目位置
            legend_opts=opts.LegendOpts(pos_right="10%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),  # 添加趋势线
            toolbox_opts=opts.ToolboxOpts(pos_left='right'),
        )

        # 基准净值曲线
        line2 = Line()
        line2.add_xaxis(data_set['trade_date'].tolist())
        line2.add_yaxis('基准净值', data_set.benchmark.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line2.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="10%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        # 叠加净值曲线和基准净值曲线
        line1.overlap(line2)
        return line1

    # 绘制IC及IC MA曲线图
    def ic_line(self, data_set, set_name):
        line1 = Line(init_opts=opts.InitOpts(width=self.figure_width, height=self.figure_height, theme=ThemeType.DARK))
        line1.add_xaxis(data_set['trade_date'].tolist())
        line1.add_yaxis('IC', data_set.ic.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line1.set_series_opts(label_opts=opts.LabelOpts(is_show=True))  # 是否显示数据标签
        line1.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                is_scale=True,
                axislabel_opts=opts.LabelOpts(rotate=60),
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                split_number=20,
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=False,
                axislabel_opts=opts.LabelOpts(formatter="{value}"),
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(type_='inside', xaxis_index=[0, 0]),  # 内部滑动
                opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1]),
            ],
            title_opts=opts.TitleOpts(title='%s IC图' % set_name, pos_left='45%'),  # 题目位置
            legend_opts=opts.LegendOpts(pos_right="10%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),  # 添加趋势线
            toolbox_opts=opts.ToolboxOpts(pos_left='right'),
        )

        # IC MA
        line2 = Line()
        line2.add_xaxis(data_set['trade_date'].tolist())
        line2.add_yaxis('IC MA', data_set.ic_ma.round(2).to_list(), yaxis_index=0, is_smooth=True)
        line2.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="10%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )

        line1.overlap(line2)
        return line1

    # 因子分布直方图
    def factor_hist(self):
        hist, bin_edges = np.histogram(data_set['fct'], bins=100)
        bar = (
            Bar(init_opts=opts.InitOpts(width=self.figure_width, height=self.figure_height, theme=ThemeType.DARK))
            .add_xaxis([str(x) for x in bin_edges[:-1]])
            .add_yaxis("数量", [float(x) for x in hist], category_gap=0)  # 直方图柱与柱之间是否有间隔
            .set_global_opts(
                title_opts=opts.TitleOpts(title="因子分布直方图", pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False),  # 标签是否显示
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-60))
            )
        )
        return bar

    # 因子预测收益率直方图
    def predict_return_hist(self):
        hist, bin_edges = np.histogram(data_set['y_hat'], bins=100)
        bar = (
            Bar(init_opts=opts.InitOpts(width=self.figure_width, height=self.figure_height, theme=ThemeType.DARK))
            .add_xaxis([str(x) for x in bin_edges[:-1]])
            .add_yaxis("数量", [float(x) for x in hist], category_gap=0)  # 直方图柱与柱之间是否有间隔
            .set_global_opts(
                title_opts=opts.TitleOpts(title="因子预测收益率直方图", pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False),  # 标签是否显示
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-60))
            )
        )
        return bar

    # 策略净值可视化
    def plot(self):

        # 回测指标统计表
        table = self.statistical_table()

        # 各个数据集的净值曲线图
        train_set_nav_line = self.nav_line(self.data_set, '全样本')

        # 总体IC图
        ic_line = self.ic_line(self.data_set, '全样本')

        # 因子分布直方图
        factor_hist = self.factor_hist()

        # 因子预测收益率直方图
        predict_return_hist = self.predict_return_hist()

        tab = Tab()
        tab.add(table, '回测指标统计')
        tab.add(train_set_nav_line, '样本内净值曲线图')
        tab.add(ic_line, '总体IC序列图')
        tab.add(factor_hist, '因子分布直方图')
        tab.add(predict_return_hist, '因子预测收益率直方图')
        tab.render('因子回测报告.html')

# 函数入口
data_set, indicators_frame = analyse(concatenated_df, '所有集合')
draw = draw_picture(data_set, indicators_frame)
draw.plot()