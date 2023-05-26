import warnings
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness
from sklearn import metrics as me
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
import time
warnings.filterwarnings('ignore')


def prepross(frequency, n_days):
    df = pd.read_pickle('./510050_pickle.pkl')
    t = 1
    # 把dataframe中的etime转化为datetime类型
    df['etime'] = pd.to_datetime(df['etime'])
    # 经过这一步处理后，今天的数据对应着明天的涨幅，比如2-23对应着24日的涨幅
    df['return'] = df['close'].pct_change(periods=t).shift(-t)
    # 删除任何行为nan的值
    df.dropna(axis=0, how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 返回列名['open', 'high', 'low', 'close', 'volume', 'return']
    fields = df.columns[1:]

    # 把每一列的值都变成浮点数float，这一步非常重要，因为gplearn只能处理float
    for col in fields:
        df[col] = df[col].values.astype('float')

    x_train = df.drop(columns=['etime', 'return']).to_numpy()
    y_train = df['return'].values
    return np.nan_to_num(x_train), np.nan_to_num(y_train)


def my_metric(y, y_pred, w):
    # 返回的是y和y_pred之间相似度的得分
    return me.normalized_mutual_info_score(y, y_pred)


def proliferate():
    # func_1是gp自带的，func_2是自己定义的，user_func等于func1加上func2
    func_1 = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan']
    func_2 = ['tanh', 'elu', 'TA_HT_TRENDLINE', 'TA_HT_DCPERIOD', 'TA_HT_DCPHASE', 'TA_SAR', 'TA_BOP', 'TA_AD',
              'TA_OBV', 'TA_TRANGE']
    user_func = func_1 + func_2

    # 调用make_fitness函数，并传入3个参数，function为my_metric
    # 这行代码返回的是 _Fitness(function=function,greater_is_better=greater_is_better)这样一个类
    user_metric = make_fitness(function=my_metric, greater_is_better=True, wrap=False)
    # 在genetic.SymbolicTransformer里，实例化一个SymbolicTransformer类
    ST_gplearn = SymbolicTransformer(population_size=200,
                                     hall_of_fame=80,
                                     n_components=50,
                                     generations=2,
                                     tournament_size=30,
                                     const_range=None,  # (-1, 1),  # critical
                                     init_depth=(1, 6),
                                     function_set=user_func,
                                     metric='spearman',
                                     # 传入的是_Fitness(function=function,greater_is_better=greater_is_better)
                                     # 在fitness函数中，
                                      #  metric='spearman', # pearson相关系数：思考的深度不够
                                     parsimony_coefficient=0.005,
                                     p_crossover=0.9,
                                     p_subtree_mutation=0.01,
                                     p_hoist_mutation=0.01,
                                     p_point_mutation=0.01,
                                     p_point_replace=0.4,
                                     feature_names=['open', 'high', 'low', 'close', 'volume'],  # 注意这里必须有feature_names
                                     n_jobs=-1,
                                     random_state=1)
    ST_gplearn.fit(x_train, y_train)
    best_programs = ST_gplearn._best_programs
    best_programs_dict = {}

    for bp in best_programs:
        factor_name = 'alpha_' + str(best_programs.index(bp) + 1)
        best_programs_dict[factor_name] = {'fitness': bp.fitness_, 'expression': str(bp), 'depth': bp.depth_,
                                           'length': bp.length_}

    best_programs_frame = pd.DataFrame(best_programs_dict).T
    best_programs_frame = best_programs_frame.sort_values(by='fitness', axis=0, ascending=False)
    best_programs_frame = best_programs_frame.drop_duplicates(subset=['expression'], keep='first')

    print(best_programs_frame)


if __name__ == '__main__':
    x_train, y_train = prepross(1, 1)
    proliferate()

