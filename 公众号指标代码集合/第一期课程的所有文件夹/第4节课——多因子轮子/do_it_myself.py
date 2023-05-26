from tkinter import X
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

'''
========================================================================================================================
生成数据表格（列名：etime、tdate、close）
========================================================================================================================
'''


class Backtest:
    def __init__(self, bgn_date, end_date, file):
        file['etime'] = pd.to_datetime(file['etime'])
        # 新增一列，让label=-1
        file['label'] = '-1'
        # 新增一列tdate,这一段代码的作用是什么？
        file['tdate'] = pd.to_datetime(file['etime']).dt.date
        # 根据开始日期和结束日期截取数据
        bgn_date = pd.to_datetime(bgn_date)
        end_date = pd.to_datetime(end_date)
        for i in range(0, len(file), 1):
            if (bgn_date <= file.at[i, 'etime']) and (file.at[i, 'etime'] <= end_date):
                file['label'] = '1'
        # 筛选然后重置数据，重置索引
        file = file[file['label'] == '1']
        file = file.reset_index(drop=True)
        # 截取etime、tdate和close
        etime_close_data = file['etime', 'tdate', 'close']
        self.etime_close_data = etime_close_data.reset_index(drop=True)

    '''
    ========================================================================================================================
    单因子分析框架
    ========================================================================================================================
    '''
    def backtest(self, original_data):
        pass


if __name__ == '__main__':
    file_name = input('请输入文件名')
    # 默认全都是xlsx文档
    read_file_path = './' + file_name + '.xlsx'
    file = pd.read_excel(read_file_path)

