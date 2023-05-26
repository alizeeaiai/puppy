# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:40:38 2022
E:\spyder_code_fold\factor_optimization_grid_search_based.py
@author: P15
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from tracemalloc import start
import warnings
from backtest import backtest
import time
import numpy as np
import pandas as pd



'''
============================================================
根据不同频率的close（收盘价）生成MACD原始数据表格
============================================================
'''
def generate_etime_close_data_divd_time(bgn_date, end_date, index_code, frequency): # 2022-11-25 edition
    # 读取数据
    read_file_path = 'E:/Factor_Work_K/7_gplearn_tiger/' + index_code + '_' + frequency + '.xlsx'
    kbars = pd.read_excel(read_file_path)
    kbars['tdate'] = pd.to_datetime(kbars['etime']).dt.date # 20221126优化
    kbars['etime'] = pd.to_datetime(kbars['etime'])
    kbars['label'] = '-1'
    # 根据区间开始和结束日期截取数据
    bgn_date = pd.to_datetime(bgn_date)
    end_date = pd.to_datetime(end_date)
    for i in range(0, len(kbars), 1): # .strftime('%Y-%m-%d %H:%M:%S')
        if (bgn_date <= kbars.at[i, 'etime']) and (kbars.at[i, 'etime'] <= end_date):
            kbars.at[i, 'label'] = '1'

    # 筛选数据并重置索引
    kbars = kbars[kbars['label'] == '1']
    kbars = kbars.reset_index(drop=True)
    etime_close_data = kbars[['etime', 'tdate', 'close']]
    etime_close_data = etime_close_data.reset_index(drop=True)

    return etime_close_data

  # 注意这里只传输进去close就可以，在backtest函数里面会计算return
'''
============================================================
网格搜索核心函数
============================================================
'''
def iter_func(params):

    freq, col_name, fct_series, data = params

    data['fct'] = fct_series.values # data['fct']为对应因子的数据
    print(data)
    # ind_frame = backtest(original_data=data, index_code='510050', frequency=freq, n_days=1)
    # print('frequency: {}\nfct_name: {}\n'.format(freq, col_name))
    # print(ind_frame)
    # print('\n')
    # print('夏普比率（样本外）：{}\n\n'.format(ind_frame.loc['样本外', '夏普比率']))
    # param_str = col_name
    # ind_frame['params'] = param_str
    #
    # return ind_frame

'''
============================================================
主程序
============================================================
'''
if __name__ == '__main__': 

    start_time = time.time()
    final_frame = pd.DataFrame()
    file_path = 'E:/Factor_Work_K/9_15_and_30_mins_models/1_data_fcts_raw_standardized/fct_gp_161_0213_stdzd_all.csv'  #
    warnings.filterwarnings('ignore') ## vol_price_feed_data_1126
    job_num = 16  # 设置并行核数
    freq_val = '15'

    original_frame = generate_etime_close_data_divd_time(bgn_date='2005-02-23 09:45:00', end_date='2022-11-29 15:00:00', index_code='510050', frequency=freq_val)
    fct_file = pd.read_csv(file_path, index_col=0)

    inputs = []
    for fct_name in fct_file.columns:
        print(fct_name)
        inputs.append((freq_val, fct_name, fct_file[fct_name], original_frame))

    with ProcessPoolExecutor(max_workers=job_num) as executor:
        results = {executor.submit(iter_func, param): param for param in inputs}
        # for r in as_completed(results):
        #     try:
        #         final_frame = pd.concat([final_frame, r.result()])
        #     except Exception as exception:
        #         print(exception)

    # final_frame.to_csv('E:/Factor_Work_K/9_15_and_30_mins_models/1_data_fcts_raw_standardized/result_161_stdzdall_K.csv', encoding='utf-8-sig') # change
    # end_time = time.time()
    # print('Time cost:====', end_time - start_time)
    
