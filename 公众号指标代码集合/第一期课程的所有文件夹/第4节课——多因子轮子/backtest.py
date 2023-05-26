import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

# 生成原始数据表格
def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency):
    # 读取数据
    read_file_path = './' + index_code + '_' + frequency + '.xlsx'
    kbars = pd.read_excel(read_file_path) # excel读取数据
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date # 用dt.date提取date
    kbars['etime'] = pd.to_datetime(kbars['etime']) # 转化为时间类型数据
    kbars['label'] = '-1'
    # 根据区间开始和结束日期截取数据
    bgn_date = pd.to_datetime(bgn_date)
    end_date = pd.to_datetime(end_date)
    for i in range(0, len(kbars), 1):  # 把需要的时间数据label标记为1
        if (bgn_date <= kbars.at[i, 'etime']) and (kbars.at[i, 'etime'] <= end_date):
            kbars.at[i, 'label'] = '1'

    # 这一步只把label=1的数据提取出来 -1的不要了 那是我们不需要的时间范围
    kbars = kbars[kbars['label'] == '1']
    # 重置索引
    kbars = kbars.reset_index(drop=True)
    # 提取出来我们需要的行，etime、tdate和close，赋值给etime_close_data
    etime_close_data = kbars[['etime', 'tdate', 'close']]
    etime_close_data = etime_close_data.reset_index(drop=True)

    return etime_close_data

def backtest(original_data, index_code, frequency, n_days):

    final_frame = original_data[['tdate', 'etime', 'close', 'fct']].dropna(axis=0).reset_index(drop=True)

    if frequency == '15':
        t_delta = int(1 * n_days)
    else:
        t_delta = int(int(240 / int(frequency)) * n_days)

    # 用shift方式计算收益率
    final_frame['close_1'] = final_frame['close'].shift(-1)
    final_frame['ret'] = (final_frame['close_1'] / final_frame['close']) - 1
    final_frame = final_frame.dropna(axis=0).reset_index(drop=True)
    # 提取数据,把etime，close，fct和ret提取出来
    data_for_model = final_frame[['etime', 'close', 'fct', 'ret']]
    # 这行代码是划分数据集，截取2016年12月30的收盘价为训练机，这里拿到了index
    train_set_end_index = data_for_model[(data_for_model['etime'].dt.year == 2016) & (data_for_model['etime'].dt.month == 12) & (data_for_model['etime'].dt.day == 30) & (data_for_model['etime'].dt.hour == 15)].index.values[0]
    # reshape(-1.1)让X_train成为二维数组
    X_train = data_for_model.loc[: train_set_end_index, 'fct'].values.reshape(-1, 1)
    y_train = data_for_model.loc[: train_set_end_index, 'ret'].values.reshape(-1, 1)
    X_test = data_for_model.loc[train_set_end_index + 1:, 'fct'].values.reshape(-1, 1)
    # 截取etime,把etime变成一个列表
    etime_train = data_for_model.loc[: train_set_end_index, 'etime'].values
    etime_test = data_for_model.loc[train_set_end_index + 1:, 'etime'].values
    etime_train_test = data_for_model.loc[:, 'etime'].values

    # 偏置要加上
    model = LinearRegression(fit_intercept=True)
    # 把x_train也就是factor和y_train也就是return扔到模型里
    model.fit(X_train, y_train)

    # 把测试集丢到模型里，预测y_test_hat 预测测试集的收益率
    y_test_hat = model.predict(X_test)
    y_test_hat = [i[0] for i in y_test_hat]

    # 把训练集丢到模型里，得到训练集合上预测的收益率
    y_train_hat = model.predict(X_train)
    # 把ndarray转化成list
    y_train_hat = [i[0] for i in y_train_hat]

    '''2：测算持仓净值（训练集）'''

    # 截取训练集的开始和结束日期,2005-2016是训练集
    begin_date_train = pd.to_datetime(str(etime_train[0])).strftime('%Y-%m-%d %H:%M:%S')
    end_date_train = pd.to_datetime(str(etime_train[-1])).strftime('%Y-%m-%d %H:%M:%S')

    # 初始化训练集的开始和结束日期
    ret_frame_train_total = generate_etime_close_data_divd_time(begin_date_train, end_date_train, index_code, frequency)
    # 这里截取到的是开始和结束的index
    start_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[0]].index.values[0]
    end_index = ret_frame_train_total[ret_frame_train_total['etime'] == etime_train[-1]].index.values[0]
    # 这一步是一个检查作用，是为了check是不是和前面的数据一样
    ret_frame_train_total = ret_frame_train_total.loc[start_index: end_index, :].reset_index(drop=True)

    # 这里的0.0005和0.01应该是2个参数，这里进入了仓位映射环节。把收益率分一分，做成仓位映射
    ret_frame_train_total['position'] = [(i / 0.0005) * 0.01 for i in y_train_hat]
    ret_frame_train_total['position'] = [1 for i in y_train_hat]

    # 如果position>1，就搞成1。如果position<-1，就搞成-1。毕竟仓位不能无限大，无限小。
    for i in range(0, len(ret_frame_train_total), 1):
        if ret_frame_train_total.at[i, 'position'] > 1:
            ret_frame_train_total.at[i, 'position'] = 1
        elif ret_frame_train_total.at[i, 'position'] < -1:
            ret_frame_train_total.at[i, 'position'] = -1

    ret_frame_train = ret_frame_train_total


    # 1：初始化持仓净值

    ret_frame_train.loc[0, '持仓净值'] = 1
    print(ret_frame_train)
    # 2：分周期测算持仓净值
    for i in range(0, len(ret_frame_train), 1):

        # 计算持仓净值
        if i == 0 or ret_frame_train.at[i - 1, 'position'] == 0:
            ret_frame_train.at[i, '持仓净值'] = 1
        else:
            close_2 = ret_frame_train.at[i, 'close']
            close_1 = ret_frame_train.at[i - 1, 'close']
            position = abs(ret_frame_train.at[i - 1, 'position'])

            if ret_frame_train.at[i - 1, 'position'] > 0:
                ret_frame_train.at[i, '持仓净值'] = 1.0 * (close_2 / close_1)
            elif ret_frame_train.at[i - 1, 'position'] < 0:
                ret_frame_train.at[i, '持仓净值'] = 1.0 * (1 - (close_2 / close_1 - 1))

    # 3：滚动测算累计持仓净值
    ret_frame_train.loc[0, '持仓净值（累计）'] = 1
    for i in range(1, len(ret_frame_train), 1):
        ret_frame_train.at[i, '持仓净值（累计）'] = ret_frame_train.at[i - 1, '持仓净值（累计）'] * ret_frame_train.at[i, '持仓净值']

    '''3：测算持仓净值（测试集）'''

    begin_date_test = pd.to_datetime(str(etime_test[0])).strftime('%Y-%m-%d %H:%M:%S')
    end_date_test = pd.to_datetime(str(etime_test[-1])).strftime('%Y-%m-%d %H:%M:%S')

    ret_frame_test_total = generate_etime_close_data_divd_time(begin_date_test, end_date_test, index_code, frequency)

    start_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[0]].index.values[0]
    end_index = ret_frame_test_total[ret_frame_test_total['etime'] == etime_test[-1]].index.values[0]

    ret_frame_test_total = ret_frame_test_total.loc[start_index: end_index, :].reset_index(drop=True)
    ret_frame_test_total['position'] = [(i / 0.0005) * 0.01 for i in y_test_hat]

    for i in range(0, len(ret_frame_test_total), 1):
        if ret_frame_test_total.at[i, 'position'] > 1:
            ret_frame_test_total.at[i, 'position'] = 1
        elif ret_frame_test_total.at[i, 'position'] < -1:
            ret_frame_test_total.at[i, 'position'] = -1

    ret_frame_test = ret_frame_test_total
    ret_frame_test = ret_frame_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引
    ret_frame_test.loc[0, '持仓净值'] = 1

    # 2：分周期测算持仓净值
    for i in range(0, len(ret_frame_test), 1):

        # 计算持仓净值
        if i == 0 or ret_frame_test.at[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
            ret_frame_test.at[i, '持仓净值'] = 1
        else:
            close_2 = ret_frame_test.at[i, 'close']
            close_1 = ret_frame_test.at[i - 1, 'close']
            position = abs(ret_frame_test.at[i - 1, 'position'])  # 获取仓位大小（上一周期）

            if ret_frame_test.at[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
                ret_frame_test.at[i, '持仓净值'] = 1.0 * (close_2 / close_1)
            elif ret_frame_test.at[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
                ret_frame_test.at[i, '持仓净值'] = 1.0 * (1 - (close_2 / close_1 - 1))

    # 3：滚动测算累计持仓净值
    ret_frame_test.loc[0, '持仓净值（累计）'] = 1
    for i in range(1, len(ret_frame_test), 1):
        ret_frame_test.at[i, '持仓净值（累计）'] = ret_frame_test.at[i - 1, '持仓净值（累计）'] * ret_frame_test.at[i, '持仓净值']

    # 重新搞一个df，设置以下几个列：etime、fct_value、y_hat、return_real等等
    out_data_test = pd.DataFrame()
    out_data_test['etime'] = etime_test
    out_data_test['fct_value'] = X_test
    out_data_test['y_hat'] = y_test_hat
    out_data_test['return_real'] = data_for_model.loc[train_set_end_index + 1:, 'ret'].values.reshape(-1, 1)
    out_data_test['position'] = ret_frame_test['position']
    out_data_test['net'] = ret_frame_test['持仓净值']
    out_data_test['accum'] = ret_frame_test['持仓净值（累计）']

    '''4：测算持仓净值（训练集 + 测试集）'''

    begin_date_train_test = pd.to_datetime(str(etime_train_test[0])).strftime('%Y-%m-%d %H:%M:%S')
    end_date_train_test = pd.to_datetime(str(etime_train_test[-1])).strftime('%Y-%m-%d %H:%M:%S')

    ret_frame_train_test_total = generate_etime_close_data_divd_time(begin_date_train_test, end_date_train_test,
                                                                     index_code, frequency)

    start_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[0]].index.values[0]
    end_index = ret_frame_train_test_total[ret_frame_train_test_total['etime'] == etime_train_test[-1]].index.values[0]

    ret_frame_train_test_total = ret_frame_train_test_total.loc[start_index: end_index, :].reset_index(
        drop=True)
    ret_frame_train_test_total['position'] = [(i / 0.0005) * 0.01 for i in y_train_hat] + [(i / 0.0005) * 0.01 for i in y_test_hat]

    ret_frame_train_test_total['fct'] = [i[0] for i in X_train] + [i[0] for i in X_test]  # 添加因子值列

    for i in range(0, len(ret_frame_train_test_total), 1):
        if ret_frame_train_test_total.at[i, 'position'] > 1:
            ret_frame_train_test_total.at[i, 'position'] = 1
        elif ret_frame_train_test_total.at[i, 'position'] < -1:
            ret_frame_train_test_total.at[i, 'position'] = -1

    ret_frame_train_test = ret_frame_train_test_total
    ret_frame_train_test = ret_frame_train_test.dropna(axis=0).reset_index(drop=True)  # 去除空值并重置索引

    # 1：初始化持仓净值
    ret_frame_train_test.loc[0, '持仓净值'] = 1

    # 2：分周期测算持仓净值
    for i in range(0, len(ret_frame_train_test), 1):

        # 计算持仓净值
        if i == 0 or ret_frame_train_test.at[i - 1, 'position'] == 0:  # 如果是第一个时间步或前一个区间的结束时刻为空仓状态
            ret_frame_train_test.at[i, '持仓净值'] = 1
        else:
            close_2 = ret_frame_train_test.at[i, 'close']
            close_1 = ret_frame_train_test.at[i - 1, 'close']
            position = abs(ret_frame_train_test.at[i - 1, 'position'])  # 获取仓位大小（上一周期）

            if ret_frame_train_test.at[i - 1, 'position'] > 0:  # 如果上一周期开的是多仓
                ret_frame_train_test.at[i, '持仓净值'] = 1.0 * (close_2 / close_1)
            elif ret_frame_train_test.at[i - 1, 'position'] < 0:  # 如果上一周期开的是空仓
                ret_frame_train_test.at[i, '持仓净值'] = 1.0 * (1 - (close_2 / close_1 - 1))

    # 3：滚动测算累计持仓净值
    ret_frame_train_test.loc[0, '持仓净值（累计）'] = 1
    for i in range(1, len(ret_frame_train_test), 1):
        ret_frame_train_test.at[i, '持仓净值（累计）'] = ret_frame_train_test.at[i - 1, '持仓净值（累计）'] * \
                                                  ret_frame_train_test.at[i, '持仓净值']

    '''
       ========================================================================================================================
       PART 2：单因子风险指标测算
       ========================================================================================================================
       '''

    '''0：设置无风险利率'''
    fixed_return = 0

    '''1：初始化'''
    indicators_frame = pd.DataFrame() # 新建一个df
    year_list = [i for i in ret_frame_train_test['etime'].dt.year.unique()]  # 获取训练集+测试集里的年份数据，获取唯一的值
    indicators_frame['年份'] = year_list + ['样本内', '样本外', '总体']
    indicators_frame = indicators_frame.set_index('年份')

    '''2：计算风险指标（总体）'''
    start_index = ret_frame_train_test.index[0]  # 获取总体的起始索引
    end_index = ret_frame_train_test.index[-1]  # 获取总体的结束索引

    # 1：总收益
    net_value_2 = ret_frame_train_test.loc[end_index, '持仓净值（累计）']
    net_value_1 = ret_frame_train_test.loc[start_index, '持仓净值（累计）']
    total_return = net_value_2 / net_value_1 - 1

    indicators_frame.loc['总体', '总收益'] = total_return

    # 2：年化收益率
    date_list = [i for i in ret_frame_train_test['etime'].dt.date.unique()]
    run_day_length = len(date_list)
    # 用math.pow函数计算年化收益率
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc['总体', '年化收益'] = annual_return

    # 3：夏普比率、年化波动率
    net_asset_value_list = []
    # 拿到ret_frame_train_test里的'每个日期'。先对日期进行groupby，然后只拿一个出来，把这些日期的index塞进一个列表
    net_asset_value_index = [i for i in ret_frame_train_test.groupby(['tdate']).tail(1).index]
    # 对index的列表进行遍历
    for date_index in net_asset_value_index:
        # 找到每个交易日结束的累计持仓净值
        net_asset_value = ret_frame_train_test.loc[date_index, '持仓净值（累计）']
        # 把持仓净值装到net_asset_value_list中
        net_asset_value_list.append(net_asset_value)

    # 新建一饿df，把date_list和持仓净值装到这个df里
    net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})
    # 在df的第一行添加一个daily_log_return一列
    net_asset_value_frame.loc[0, 'daily_log_return'] = 0

    # 对所有行数进行遍历。一共有n个交易日，这个代码喜欢写步长，这里步长为1
    for i in range(1, len(net_asset_value_frame), 1):
        # 为每一行添加一个新列['daily_log_return']，然后计算每日对数收益率
        net_asset_value_frame.at[i, 'daily_log_return'] = math.log(net_asset_value_frame.at[i, 'nav']) - math.log(
            net_asset_value_frame.at[i - 1, 'nav'])
    # 计算年华波动率
    annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
    sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

    indicators_frame.loc['总体', '年化波动率'] = annual_volatility
    indicators_frame.loc['总体', '夏普比率'] = sharpe_ratio

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）

    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (
        np.maximum.accumulate(net_asset_value_list)))
    if mdd_end_index == 0:
        return 0
    mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日

    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日

    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (
    net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc['总体', '最大回撤率'] = maximum_drawdown
    indicators_frame.loc['总体', '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc['总体', '最大回撤结束日'] = mdd_end_date
    # print(indicators_frame, ' -------checked line 512') # checked at 202211262310

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc['总体', '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(ret_frame_train_test)  # 计算总交易次数

    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

    for i in range(1, len(ret_frame_train_test), 1):
        delta_value = ret_frame_train_test.at[i, '持仓净值（累计）'] - ret_frame_train_test.loc[
            i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
        win_lose_frame.at[i, 'delta_value'] = delta_value
        if delta_value > 0:
            win_times = win_times + 1

    gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
    loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

    winning_rate = win_times / total_trading_times  # 计算胜率
    gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

    indicators_frame.loc['总体', '总交易次数'] = total_trading_times
    indicators_frame.loc['总体', '胜率'] = winning_rate
    indicators_frame.loc['总体', '盈亏比'] = gain_loss_ratio
    # print(indicators_frame, '=----checked line 540') # checked at 202211262310

    '''3：计算风险指标（分年度）'''
    for year in year_list:
        data_demo = ret_frame_train_test[ret_frame_train_test['etime'].dt.year == year]  # 提取数据
        data_demo = data_demo.reset_index(drop=True)  # 重置索引
        data_demo['持仓净值（累计）'] = data_demo['持仓净值（累计）'] / data_demo.loc[0, '持仓净值（累计）']  # 缩放区间内部累计持仓净值

        start_index = data_demo.index[0]  # 获取当年的起始索引
        end_index = data_demo.index[-1]  # 获取当年的结束索引

        # 1：总收益
        net_value_2 = data_demo.loc[end_index, '持仓净值（累计）']
        net_value_1 = data_demo.loc[start_index, '持仓净值（累计）']
        total_return = net_value_2 / net_value_1 - 1

        indicators_frame.loc[year, '总收益'] = total_return

        # 2：年化收益率
        date_list = [i for i in data_demo['etime'].dt.date.unique()]
        run_day_length = len(date_list)  # 计算策略运行天数
        annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

        indicators_frame.loc[year, '年化收益'] = annual_return

        # 3：夏普比率、年化波动率
        net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
        net_asset_value_index = [i for i in data_demo.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

        for date_index in net_asset_value_index:
            net_asset_value = data_demo.loc[date_index, '持仓净值（累计）']
            net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

        net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
        net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

        for i in range(1, len(net_asset_value_frame), 1):
            net_asset_value_frame.at[i, 'daily_log_return'] = math.log(net_asset_value_frame.at[i, 'nav']) - math.log(
                net_asset_value_frame.at[i - 1, 'nav'])  # 计算对数收益率（日度）

        annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
        sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

        indicators_frame.loc[year, '年化波动率'] = annual_volatility
        indicators_frame.loc[year, '夏普比率'] = sharpe_ratio

        # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
        mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (
            np.maximum.accumulate(net_asset_value_list)))
        if mdd_end_index == 0:
            return 0
        mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日

        mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
        mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日

        maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (
        net_asset_value_list[mdd_start_index])  # 计算最大回撤率

        indicators_frame.loc[year, '最大回撤率'] = maximum_drawdown
        indicators_frame.loc[year, '最大回撤起始日'] = mdd_start_date
        indicators_frame.loc[year, '最大回撤结束日'] = mdd_end_date

        # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
        calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

        indicators_frame.loc[year, '卡尔玛比率'] = calmar_ratio

        # 6：总交易次数、交易胜率、交易盈亏比
        total_trading_times = len(data_demo)  # 计算总交易次数

        win_times = 0  # 初始化盈利次数
        win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

        for i in range(1, len(data_demo), 1):
            delta_value = data_demo.at[i, '持仓净值（累计）'] - data_demo.at[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
            win_lose_frame.at[i, 'delta_value'] = delta_value
            if delta_value > 0:
                win_times = win_times + 1

        gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
        loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

        winning_rate = win_times / total_trading_times  # 计算胜率
        gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

        indicators_frame.loc[year, '总交易次数'] = total_trading_times
        indicators_frame.loc[year, '胜率'] = winning_rate
        indicators_frame.loc[year, '盈亏比'] = gain_loss_ratio

    '''4：计算风险指标（样本内）'''
    start_index = ret_frame_train.index[0]  # 获取训练集的起始索引
    end_index = ret_frame_train.index[-1]  # 获取训练集的结束索引

    # 1：总收益
    net_value_2 = ret_frame_train.loc[end_index, '持仓净值（累计）']
    net_value_1 = ret_frame_train.loc[start_index, '持仓净值（累计）']
    total_return = net_value_2 / net_value_1 - 1

    indicators_frame.loc['样本内', '总收益'] = total_return

    # 2：年化收益率
    date_list = [i for i in ret_frame_train['etime'].dt.date.unique()]
    run_day_length = len(date_list)  # 计算策略运行天数
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc['样本内', '年化收益'] = annual_return

    # 3：夏普比率、年化波动率
    net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
    net_asset_value_index = [i for i in ret_frame_train.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

    for date_index in net_asset_value_index:
        net_asset_value = ret_frame_train.loc[date_index, '持仓净值（累计）']
        net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

    net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
    net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

    for i in range(1, len(net_asset_value_frame), 1):
        net_asset_value_frame.at[i, 'daily_log_return'] = math.log(net_asset_value_frame.at[i, 'nav']) - math.log(
            net_asset_value_frame.at[i - 1, 'nav'])  # 计算对数收益率（日度）

    annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
    sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

    indicators_frame.loc['样本内', '年化波动率'] = annual_volatility
    indicators_frame.loc['样本内', '夏普比率'] = sharpe_ratio

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (
        np.maximum.accumulate(net_asset_value_list)))
    if mdd_end_index == 0:
        return 0
    mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日

    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日

    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (
    net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc['样本内', '最大回撤率'] = maximum_drawdown
    indicators_frame.loc['样本内', '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc['样本内', '最大回撤结束日'] = mdd_end_date

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc['样本内', '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(ret_frame_train)  # 计算总交易次数

    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

    for i in range(1, len(ret_frame_train), 1):
        delta_value = ret_frame_train.at[i, '持仓净值（累计）'] - ret_frame_train.at[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
        win_lose_frame.at[i, 'delta_value'] = delta_value
        if delta_value > 0:
            win_times = win_times + 1

    gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
    loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

    winning_rate = win_times / total_trading_times  # 计算胜率
    gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

    indicators_frame.loc['样本内', '总交易次数'] = total_trading_times
    indicators_frame.loc['样本内', '胜率'] = winning_rate
    indicators_frame.loc['样本内', '盈亏比'] = gain_loss_ratio
    """
    以下是样本外计算
    ---————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    """
    '''5：计算风险指标（样本外）'''
    start_index = ret_frame_test.index[0]  # 获取测试集的起始索引
    end_index = ret_frame_test.index[-1]  # 获取测试集的结束索引

    # 1：总收益
    net_value_2 = ret_frame_test.loc[end_index, '持仓净值（累计）']
    net_value_1 = ret_frame_test.loc[start_index, '持仓净值（累计）']
    total_return = net_value_2 / net_value_1 - 1

    indicators_frame.loc['样本外', '总收益'] = total_return

    # 2：年化收益率
    date_list = [i for i in ret_frame_test['etime'].dt.date.unique()]
    run_day_length = len(date_list)  # 计算策略运行天数
    annual_return = math.pow(1 + total_return, 252 / run_day_length) - 1

    indicators_frame.loc['样本外', '年化收益'] = annual_return

    # 3：夏普比率、年化波动率
    net_asset_value_list = []  # 初始化累计持仓净值列表（日度）
    net_asset_value_index = [i for i in ret_frame_test.groupby(['tdate']).tail(1).index]  # 获取每日的结束索引

    for date_index in net_asset_value_index:
        net_asset_value = ret_frame_test.loc[date_index, '持仓净值（累计）']
        net_asset_value_list.append(net_asset_value)  # 附加每日结束时对应的累计持仓净值

    net_asset_value_frame = pd.DataFrame({'tdate': date_list, 'nav': net_asset_value_list})  # 构建日度累计持仓净值表格
    net_asset_value_frame.loc[0, 'daily_log_return'] = 0  # 初始化对数收益率（日度）

    for i in range(1, len(net_asset_value_frame), 1):
        net_asset_value_frame.at[i, 'daily_log_return'] = math.log(net_asset_value_frame.at[i, 'nav']) - math.log(
            net_asset_value_frame.at[i - 1, 'nav'])  # 计算对数收益率（日度）

    annual_volatility = math.sqrt(252) * net_asset_value_frame['daily_log_return'].std()  # 计算年化波动率
    sharpe_ratio = (annual_return - fixed_return) / annual_volatility  # 计算夏普比率

    indicators_frame.loc['样本外', '年化波动率'] = annual_volatility
    indicators_frame.loc['样本外', '夏普比率'] = sharpe_ratio

    # 4：最大回撤率及其对应的起止日（需要利用计算夏普比率过程中构建的日度累计持仓净值表格）
    mdd_end_index = np.argmax((np.maximum.accumulate(net_asset_value_list) - net_asset_value_list) / (
        np.maximum.accumulate(net_asset_value_list)))
    if mdd_end_index == 0:
        return 0
    mdd_end_date = net_asset_value_frame.loc[mdd_end_index, 'tdate']  # 最大回撤起始日

    mdd_start_index = np.argmax(net_asset_value_list[: mdd_end_index])
    mdd_start_date = net_asset_value_frame.loc[mdd_start_index, 'tdate']  # 最大回撤结束日

    maximum_drawdown = (net_asset_value_list[mdd_start_index] - net_asset_value_list[mdd_end_index]) / (
    net_asset_value_list[mdd_start_index])  # 计算最大回撤率

    indicators_frame.loc['样本外', '最大回撤率'] = maximum_drawdown
    indicators_frame.loc['样本外', '最大回撤起始日'] = mdd_start_date
    indicators_frame.loc['样本外', '最大回撤结束日'] = mdd_end_date

    # 5：卡尔玛比率（基于夏普比率以及最大回撤率）
    calmar_ratio = (annual_return - fixed_return) / maximum_drawdown  # 计算卡尔玛比率

    indicators_frame.loc['样本外', '卡尔玛比率'] = calmar_ratio

    # 6：总交易次数、交易胜率、交易盈亏比
    total_trading_times = len(ret_frame_test)  # 计算总交易次数

    win_times = 0  # 初始化盈利次数
    win_lose_frame = pd.DataFrame()  # 初始化盈亏表格

    for i in range(1, len(ret_frame_test), 1):
        delta_value = ret_frame_test.at[i, '持仓净值（累计）'] - ret_frame_test.at[i - 1, '持仓净值（累计）']  # 计算每次交易过程中累计持仓净值的变化量
        win_lose_frame.at[i, 'delta_value'] = delta_value
        if delta_value > 0:
            win_times = win_times + 1

    gain_amount = abs(win_lose_frame[win_lose_frame['delta_value'] > 0]['delta_value'].sum())  # 计算总盈利额
    loss_amount = abs(win_lose_frame[win_lose_frame['delta_value'] < 0]['delta_value'].sum())  # 计算总亏损额

    winning_rate = win_times / total_trading_times  # 计算胜率
    gain_loss_ratio = gain_amount / loss_amount  # 计算盈亏比

    indicators_frame.loc['样本外', '总交易次数'] = total_trading_times
    indicators_frame.loc['样本外', '胜率'] = winning_rate
    indicators_frame.loc['样本外', '盈亏比'] = gain_loss_ratio

    return indicators_frame


if __name__ == '__main__':
    freq_val = '15'
    orginal_data = generate_etime_close_data_divd_time(bgn_date='2005-02-23 09:45:00', end_date='2022-11-29 15:00:00',
                                                         index_code='510050', frequency=freq_val)
    backtest(original_data=orginal_data, index_code='510050', frequency='15', n_days=1)
