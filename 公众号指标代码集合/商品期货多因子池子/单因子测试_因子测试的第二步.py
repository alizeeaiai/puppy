import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from datetime import datetime
import talib as ta
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.options import ComponentTitleOpts
from pyecharts.globals import ThemeType
from pyecharts.components import Table
import warnings
warnings.filterwarnings('ignore')

class SingleFactorModel:

    # factor_name : 因子名称
    # result_path : 回测结果保存目录
    # price_data : 价格数据dataframe，必须包含timestamp和close字段，且不要将timestamp字段设置为Index
    # factor_data ： 因子数据dataframe，必须包含timestamp和factor字段，且不要将timestamp字段设置为Index
    # n_periods : 预测n周期之后的收益率
    # ic_periods : 滚动计算ic值的周期
    # ic_ma : 计算ic均值的周期
    # rf: 无风险收益率
    # period_N: 平均每年交易日数
    def __init__(
            self,
            factor_name,
            result_path,
            price_data,
            factor_data,
            n_periods=1,
            ic_periods=30,
            ic_ma=120,
            rf=0.04,
            period_N=252
    ):
        self.factor_name = factor_name
        self.result_path = result_path
        self.price_data = price_data.copy()
        self.factor_data = factor_data.copy()
        self.n_periods = n_periods
        self.ic_periods = ic_periods
        self.ic_ma = ic_ma
        self.rf = rf
        self.period_N = period_N

        # 设置画布大小
        self.figure_width = '1400px'
        self.figure_height = '600px'

        # 添加tdate字段
        self.price_data['tdate'] = pd.to_datetime(self.price_data['timestamp']).dt.date

        # 把timestamp变成标准时间格式
        self.factor_data['timestamp'] = pd.to_datetime(self.factor_data['timestamp'])
        self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])

        # 合并价格数据和因子数据
        self.merge_data = pd.merge(self.price_data, self.factor_data, how='left', left_on='timestamp', right_on='timestamp')
        self.merge_data = self.merge_data.dropna(axis=0).reset_index(drop=True)

        # 数据没用的列太多，删除掉这些没用的列
        self.merge_data = self.merge_data.drop(['open', 'high', 'low', 'amount', 'alpha_gp_2_004', 'alpha_gp_2_032', 'volume'], axis=1)
        # 计算未来n个周期的收益率
        self.merge_data['close_nperiods'] = self.merge_data['close'].shift(-n_periods)
        self.merge_data['return_nperiods'] = self.merge_data['close_nperiods'] / self.merge_data['close'] - 1

        # 最后一行是nan，要删掉，并且重置索引
        self.merge_data = self.merge_data.dropna(axis=0).reset_index(drop=True)
        # 去掉close_nperiods一列
        self.merge_data = self.merge_data.drop(['close_nperiods'], axis=1)
        self.merge_data['timestamp'] = pd.to_datetime(self.merge_data['timestamp'], format='%Y-%m-%d %H:%M:%S')

        # 初始化训练集和测试集
        self.train_set_bgn_date = None
        self.train_set_end_date = None
        self.test_set_bgn_date = None
        self.test_set_end_date = None
        self.train_set = None
        self.test_set = None

        self.indicatos_frame = pd.DataFrame()

    def split_train_test(self, train_set_bgn_date, train_set_end_date, test_set_bgn_date, test_set_end_date):
        self.train_set_bgn_date = datetime.strptime(train_set_bgn_date, '%Y-%m-%d %H:%M:%S')
        self.train_set_end_date = datetime.strptime(train_set_end_date, '%Y-%m-%d %H:%M:%S')
        self.test_set_bgn_date = datetime.strptime(test_set_bgn_date, '%Y-%m-%d %H:%M:%S')
        self.test_set_end_date = datetime.strptime(test_set_end_date, '%Y-%m-%d %H:%M:%S')

        self.train_set = self.merge_data[(self.merge_data['timestamp'] >= self.train_set_bgn_date) & (self.merge_data['timestamp'] <= self.train_set_end_date)]
        self.test_set = self.merge_data[(self.merge_data['timestamp'] >= self.test_set_bgn_date) & (self.merge_data['timestamp'] <= self.test_set_end_date)]

        self.train_set.reset_index(drop=True, inplace=True)
        self.test_set.reset_index(drop=True, inplace=True)

    # 训练模型
    def fit(self):
        # 初始化线性回归模型
        self.linear_model = LinearRegression(fit_intercept=True)

        # 训练集数据预测
        x_train = self.train_set[factor_name].values.reshape(-1, 1)
        y_train = self.train_set['return_nperiods'].values.reshape(-1, 1)
        self.linear_model.fit(x_train, y_train)
        y_train_hat = self.linear_model.predict(x_train)

        # 测试集数据预测
        x_test = self.test_set[factor_name].values.reshape(-1, 1)
        y_test = self.test_set[factor_name].values.reshape(-1, 1)
        y_test_hat = self.linear_model.predict(x_test)

        # 总体预测
        x_all = self.merge_data[factor_name].values.reshape(-1, 1)
        y_all = self.linear_model.predict(x_all)

        # 打印出预测结果
        self.train_set['y_hat'] = [i[0] for i in y_train_hat]
        self.test_set['y_hat'] = [i[0] for i in y_test_hat]
        self.merge_data['y_hat'] = [i[0] for i in y_all]

        # 计算IC值和IC均值
        self.train_set['ic'] = self.train_set[factor_name].rolling(self.ic_periods).corr(self.train_set['return_nperiods'])
        self.train_set['ic_ma'] = ta.MA(self.train_set['ic'], timeperiod=self.ic_ma)

        self.test_set['ic'] = self.test_set[factor_name].rolling(self.ic_periods).corr(
            self.test_set['return_nperiods'])
        self.test_set['ic_ma'] = ta.MA(self.test_set['ic'], timeperiod=self.ic_ma)

        self.merge_data['ic'] = self.merge_data[factor_name].rolling(self.ic_periods).corr(
            self.merge_data['return_nperiods'])
        self.merge_data['ic_ma'] = ta.MA(self.merge_data['ic'], timeperiod=self.ic_ma)

        print('下面是训练集合')
        print(self.train_set)
        print('下面是测试集合')
        print(self.test_set)
        print('下面是整体集合')
        print(self.merge_data)
        print('______________________________________________________________________________')

    def backtest_data_set(self, data_set,  test_model, pos_coef, pos_thd, fee_rate):
        # 传入训练集/测试集/整体集合，这里先传入整体集合
        data_set = data_set.copy()

        # 计算标的每一天的净值
        data_set['bench_mark'] = data_set.close / data_set.close[0]

        # 计算每一日的涨幅
        data_set['change_pct'] = data_set.close.pct_change(1).fillna(0)

        # 计算数据集每日预测仓位,simple_test是简单预测
        simple_test = False

        if simple_test == True:
            data_set['predict_pos'] = data_set['y_hat'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        else:
            data_set['predict_pos'] = data_set['y_hat'] * 20
            data_set['predict_pos'] = data_set['predict_pos'].clip(-1, 1)

        # 计算训数据集每日持仓，因为预测的仓位要在下一个交易日才能生效，所以需要对预测仓位进行shift
        data_set['pos'] = data_set['predict_pos'].shift(1)
        data_set = data_set.replace([np.inf, -np.inf, np.nan], 0.0)

        # 计算上周期的仓位
        data_set['last_pos'] = data_set['pos'].shift(1)
        data_set = data_set.replace([np.inf, -np.inf, np.nan], 0.0)

        # 计算本周期比上周期新增的仓位
        data_set['increased_pos'] = data_set['pos'] - data_set['last_pos']

        # 计算净值，这里的0.1是手续费
        fee_rate = 0.1
        data_set['nav'] = (1 + (data_set.last_pos + data_set.increased_pos * (1 - fee_rate)) * data_set.change_pct).cumprod()

        # 计算收益率
        data_set['last_nav'] = data_set.nav.shift(1)
        data_set['increased_nav'] = data_set['nav'] - data_set['last_nav']
        data_set['profit'] = data_set.nav.pct_change(1)

        # 去除空值
        data_set = data_set.replace([np.inf, -np.inf, np.nan], 0.0)
        return data_set

    def backtest(self, test_model, pos_coef, pos_thd, fee_rate=0):

        # 对各个数据集进行回测
        print('——————————————————————训练集Train回测结果————————————————————————')
        self.train_set = self.backtest_data_set(self.train_set, test_model, pos_coef, pos_thd, fee_rate)
        print(self.train_set)
        print('——————————————————————测试集Test回测结果————————————————————————')
        self.test_set = self.backtest_data_set(self.test_set, test_model, pos_coef, pos_thd, fee_rate)
        print(self.test_set)
        print('——————————————————————所有集合merge_data回测结果————————————————————————')
        self.merge_data = self.backtest_data_set(self.merge_data, test_model, pos_coef, pos_thd, fee_rate)
        print(self.merge_data)

    def analyse(self, data_set, set_name):
        indicator_frame = pd.DataFrame()

        # 获取交易日列表
        date_list = [i for i in data_set['timestamp'].dt.date.unique()]

        # 获取每日净值列表 对所有的tdate进行分组groupby，取最后一个的净值
        nav_daily = [i for i in data_set.groupby('tdate').tail(1).nav]
        daily_nav = pd.DataFrame({'date_list': date_list, 'nav': nav_daily})

        # 计算每日收益率
        daily_nav['return_daily'] = daily_nav.nav.pct_change(1).fillna(0)

        # 总收益率
        total_return = daily_nav.nav[daily_nav.shape[0] - 1] / daily_nav.nav[0] - 1
        # 年化收益率
        annual_return = (1 + total_return) ** (self.period_N/len(date_list)) - 1

        # 最大回撤率和起始日期 用到了cummax行数，这个函数应用于每一列，每一列的每一行返回这列所有数中的最大值
        drawdown_list = 1 - data_set.nav / data_set.nav.cummax()
        max_drawndown = max(drawdown_list)
        # 最大回测开始日期和结束日期 np.argmax返回列表最大值的index
        end_date = date_list[np.argmax(drawdown_list)]
        start_date = date_list[np.argmax(nav_daily[: np.argmax(drawdown_list)])]

        # 夏普比率


        # 卡尔玛比率 卡尔玛比率 = （年化收益率-rf）/最大回撤
        calmar_ratio = (annual_return - self.rf) / max_drawndown

        # 总交易次数
        total_trading_times = len(data_set[data_set['pos'] != 0])
        win_data = data_set[data_set['profit'] > 0]
        lose_data = data_set[data_set['profit'] < 0]

        # 胜率
        winning_rate = len(win_data) / total_trading_times

        # 平均每笔交易的盈亏比
        profit_loss_ratio = win_data['profit'].mean() / abs(lose_data['profit'].mean())

        # 总盈亏比
        total_profit_loss_ratio = win_data['increased_nav'].sum() / abs(lose_data['increased_nav'].sum())

        # IC均值
        ic = data_set['ic'].mean()

        # IR
        ir = data_set['ic'].mean() / data_set['ic'].std()

        indicator_frame.loc[set_name, '总收益率'] = total_return
        indicator_frame.loc[set_name, '年化收益率'] = annual_return
        indicator_frame.loc[set_name, '卡尔玛比率'] = calmar_ratio
        indicator_frame.loc[set_name, '最大回撤'] = max_drawndown
        indicator_frame.loc[set_name, '最大回撤开始日期'] = start_date
        indicator_frame.loc[set_name, '最大回撤结束日期'] = end_date
        indicator_frame.loc[set_name, '胜率'] = winning_rate
        indicator_frame.loc[set_name, '平均每笔交易盈亏比'] = profit_loss_ratio
        indicator_frame.loc[set_name, '总盈亏比'] = total_profit_loss_ratio
        indicator_frame.loc[set_name, 'IC'] = ic
        indicator_frame.loc[set_name, 'IR'] = ir

        return indicator_frame

    # 因子回测分析
    def show_performance(self):
        indicator_train = self.analyse(self.train_set, '样本内')
        indicator_test = self.analyse(self.test_set, '样本外')
        indicator_merge_data = self.analyse(self.merge_data, '总体')

        self.indicatos_frame = pd.DataFrame()
        self.indicatos_frame = pd.concat([self.indicatos_frame, indicator_train, indicator_test, indicator_merge_data])
        print('——————————————————下面是样本内/样本外/总体的回测结果——————————————————————')
        print(self.indicatos_frame)


if __name__ == '__main__':
    factor_name = 'alpha_gp_1_035'
    result_path = './'
    price_data = pd.read_excel('./510050_d.xlsx')
    factor_data = pd.read_csv('./processed_factor.csv')
    sing_factor_model = SingleFactorModel(factor_name, result_path, price_data, factor_data)
    sing_factor_model.split_train_test('2005-02-24 10:23:00', '2018-02-24 10:23:00', '2018-02-25 10:23:00', '2022-06-22 10:23:00')
    sing_factor_model.fit()

    test_model = 0
    pos_coef = 0.2
    pos_thd = 1
    fee_rate = 0
    # 调用backtest，获得训练集、测试集和所有集合的回测结果
    sing_factor_model.backtest(test_model, pos_coef, pos_thd, fee_rate)


    sing_factor_model.show_performance()


