import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import warnings
warnings.filterwarnings('ignore')


class FactorProcess:
    def __init__(self, factor, price):
        # 先处理price数据
        self.price = price[['timestamp', 'close']]
        # 把close变成对数收益,close这一列变为"今天的收盘价比昨天高/低了多少"
        self.price['close'] = np.log(self.price['close']).diff(1)

        # 再处理factor数据，第一步先把nan值都换成0
        factor = factor.replace([np.nan], 0)
        # 重置索引，inplace=True是直接在原对象上修改
        factor.reset_index(drop=True, inplace=True)

        #把每一天都变成当月的20日，不懂这一步是什么意思
        renewed_gap = 20
        factor['timestamp'] = pd.to_datetime(factor['timestamp']) + datetime.timedelta(days=renewed_gap)
        # 把timestamp设置为索引
        self.factor = factor.set_index('timestamp')

    def fraction_or_difference(self):
        # 如果用差值，这里要设置为True
        if_difference = True
        if if_difference == True:
            self.factor = self.factor - self.factor.shift(1)
        else:
            self.factor = (self.factor - self.factor.shift(1)) / np.abs(self.factor.shift(1))

    def smooth(self):
        # 先取到每个因子序列的标准差
        new_series_1 = pd.Series(index=self.factor.columns)
        for c in self.factor.columns:
            new_series_1[c] = self.factor[self.factor[c] != 0].loc[:, c].quantile(0.5)

        # 新建一个dataframe，这个dataframe里是原来的每个因子数据点减去1个标准差，然后取绝对值，也就是计算离差值
        new_df = (self.factor * 1.0 - new_series_1).abs()  # 每个因子的数据点减去标准差，然后取绝对值，得到离差值
        # 下面这一坨是取离差值的中位数
        new_series_2 = pd.Series(index=new_df.columns)
        for i in new_df.columns:
            new_series_2[i] = new_df[new_df[i] != 0].loc[:, i].quantile(0.5)
        # 最大值等于中位数 加上 6倍离差值的中位数
        max_fct = new_series_1 + 6 * new_series_2
        min_fct = new_series_1 - 6 * new_series_2
        # 做剪裁处理
        self.factor = self.factor.clip(min_fct, max_fct, axis=1)
        self.factor.reset_index(inplace=True)

    def join(self):
        # 再次把timestamp这一列变成datetime格式
        self.factor['timestamp'] = pd.to_datetime(self.factor['timestamp'])
        self.price['timestamp'] = pd.to_datetime(self.price['timestamp'])
        # 对每一行的timestamp,如果是星期六就加上2天，星期日就加上1天。这么做的目的是
        for i in range(0, self.factor.shape[0]):
            if self.factor.loc[:, 'timestamp'][i].weekday() == 5:
                self.factor.loc[:, 'timestamp'][i] = pd.to_datetime(self.factor.loc[:, 'timestamp'][i]) + datetime.timedelta(days=2)
            elif self.factor.loc[:, 'timestamp'][i].weekday() == 6:
                self.factor.loc[:, 'timestamp'][i] = pd.to_datetime(self.price.loc[:, 'timestamp'][i]) + datetime.timedelta(days=1)
            else:
                self.factor.loc[:, 'timestamp'][i] = self.factor.loc[:, 'timestamp'][i]
        # 把2个数据合并起来,注意这里要以self.price为对齐的对象
        merge_data = pd.merge(self.price, self.factor, on='timestamp', how='left')
        # 删除所有nan值,inplace=True代替原对象
        merge_data.dropna(subset=['close'], inplace=True)
        # 后向填充
        merge_data = merge_data.fillna(method='pad')
        # 前向填充,这里注意要先后向填充，再前向填充
        merge_data = merge_data.fillna(method='bfill')
        # 去掉close这一列
        merge_data = merge_data.drop(['close'], axis=1)
        # 把timestamp这一列设置为索引
        self.merge_data = merge_data.set_index('timestamp')

    def decay(self):
        # 因子会随着时间衰减，所以新增一列
        self.merge_data['t_ondecay'] = 0
        print(self.merge_data)
        for i in range(self.merge_data.shape[0]):
            if i == 1:
                self.merge_data['t_ondecay'] = 0
            else:
                if self.merge_data[self.merge_data.columns[0]][i] == self.merge_data[self.merge_data.columns[0]][i - 1]:
                    self.merge_data['t_ondecay'][i] = self.merge_data['t_ondecay'][i - 1] + 1
                else:
                    self.merge_data['t_ondecay'][i] = 0
        self.merge_data.to_csv('./puppy的因子处理最终数据.csv')
        # 设置alpha衰减系数，经过大量实践，alpha系数设定为0.2
        alpha = 0.2
        for i in range(0, len(self.merge_data.columns)):
            self.merge_data[self.merge_data.columns[i]] = self.merge_data[self.merge_data.columns[i]] * np.exp(-alpha * self.merge_data['t_ondecay'])
        # 把t ondecay这一列drop掉
        self.merge_data = self.merge_data.drop(['t_ondecay'], axis=1)
        self.merge_data.to_csv('./processed_factor.csv')

    def plot(self):
        plt.plot(self.merge_data, 'b', label='data')
        plt.title('GPD_for_factors')
        plt.xlabel('time')
        plt.ylabel('height')
        # loc=best表示自动选择最佳位置来添加图列
        plt.legend(loc='best')
        plt.show()

    # angel处理decay的方式
    def angel_way(self):
        # 6. decay->数值衰减处理
        decay = True
        print(self.merge_data)
        if decay:
            self.merge_data['t_ondecay'] = self.merge_data.groupby(
                (self.merge_data[self.merge_data.columns[0]] != self.merge_data[self.merge_data.columns[0]].shift()).cumsum()).cumcount()
            alpha = 0.2
            for c in self.merge_data.columns:
                self.merge_data.loc[:, c] = self.merge_data.loc[:, c] * np.exp(-alpha * self.merge_data['t_ondecay'])

            fct_name = self.merge_data.drop(['t_ondecay'], axis=1)


if __name__ == '__main__':
    factor = pd.read_csv('./因子数据_国内生产总值_同比数据.csv')
    price = pd.read_excel('./510050_d.xlsx')
    factor_process = FactorProcess(factor, price)
    factor_process.fraction_or_difference()
    factor_process.smooth()
    factor_process.join()
    factor_process.decay()



