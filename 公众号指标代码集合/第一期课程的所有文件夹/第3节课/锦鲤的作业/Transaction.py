import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime


class Transaction:

    # df_price : ohlcv数据dataframe
    # fee_rate : 开仓交易成本设置，例如：0.0005
    # trans_flag : 交易标志，1:做多，-1：做空，0：无交易
    # position : 仓位比例，1:100%多头仓位，-1:100%空头仓位
    # orderList : 记录交易信息
    def __init__(self, df_price, fee_rate=0.0005):
        self.df_price = df_price.copy()
        self.df_price = self.df_price.replace([np.inf, -np.inf, np.nan], 0.0)
        self.fee_rate = fee_rate
        self.df_price['trans_flag'] = 0
        self.df_price['position'] = 0
        self.df_price['fee_rate'] = 0
        self.orderList = []
        self.orderid = 0

    # 查询仓位情况
    # index : df_price的索引
    def queryPosition(self, index):
        return self.df_price['position'][index]

    # 执行交易
    # index : df_price的索引
    # buyOrSell : 买入还是卖出, 'buy':买入, 'sell':卖出
    # openOrClose : 开仓还是平仓, 'open': 开仓, 'close': 平仓
    # position : 仓位比例，1:100%多头仓位，-1:100%空头仓位
    # price : 成交价格
    # remark : 交易说明
    # 说明 : 开仓价格要用收盘价，不然收益率会计算不准确
    def excuteTrans(self, index, buyOrSell, openOrClose, position, price, remark):

        # 记录订单id
        if openOrClose == 'open':
            self.orderid = index

        # 记录交易记录
        self.orderList.append([
            self.df_price.index[index],
            buyOrSell,
            openOrClose,
            position,
            price,
            self.orderid,
            remark
        ])

        # 记录仓位，交易当天的持仓仍旧按照昨天的持仓来算
        if self.df_price.loc[self.df_price.index[index], 'position'] == 0:
            self.df_price.loc[self.df_price.index[index], 'position'] = self.df_price.loc[self.df_price.index[index - 1], 'position']

        # 记录仓位，开仓之后持有的仓位要从第二天开始计算
        if index < self.df_price.shape[0]-1:
            self.df_price.loc[self.df_price.index[index+1], 'position'] = position

        # 记录交易标志
        if openOrClose == 'open':

            # 记录手续费
            self.df_price.loc[self.df_price.index[index], 'fee_rate'] = self.fee_rate

            # 记录开仓标志
            if buyOrSell == 'buy':
                self.df_price.loc[self.df_price.index[index], 'trans_flag'] = 1
            else:
                self.df_price.loc[self.df_price.index[index], 'trans_flag'] = -1
        else:
            self.df_price.loc[self.df_price.index[index], 'trans_flag'] = 0

    # 保持持仓
    def keepPosition(self, index):
        if self.df_price.loc[self.df_price.index[index], 'position'] == 0:
            self.df_price.loc[self.df_price.index[index], 'position'] = self.df_price.loc[self.df_price.index[index-1], 'position']

    # 回测结束之后统计交易记录
    # start_index : 统计开始的df_price索引
    def statistics(self, start_index=0):
        df_price = self.df_price.copy()
        df_price = df_price.iloc[start_index:, :]

        # 计算标的每日涨跌幅
        df_price['change_pct'] = df_price.close.pct_change(1).fillna(0)

        # 计算策略净值
        df_price['net_value'] = (1 - df_price.fee_rate) * (1 + df_price.change_pct * df_price.position).cumprod()

        # 计算标的净值
        df_price['benchmark'] = df_price.close / df_price.close[0]

        # 交易记录
        order_list = pd.DataFrame(self.orderList, columns=['日期', '买卖', '开平', '仓位', '价格', '订单号', '备注'])

        # 交易记录按照订单号合并
        order_open = order_list[order_list['开平'] == 'open']
        order_close = order_list[order_list['开平'] == 'close']
        order_open.rename(columns={'日期': '开仓日期', '买卖': '开仓', '仓位': '开仓仓位', '价格': '开仓价格', '备注': '开仓备注'}, inplace=True)
        order_open = order_open.drop(labels=['开平'], axis=1)
        order_close.rename(columns={'日期': '平仓日期', '买卖': '平仓', '仓位': '平仓仓位', '价格': '平仓价格', '备注': '平仓备注'}, inplace=True)
        order_close = order_close.drop(labels=['开平'], axis=1)
        merge_order = pd.merge(order_open, order_close, how='left', left_on='订单号', right_on='订单号')

        # 返回全过程数据以及交易记录
        return df_price, order_list, merge_order
