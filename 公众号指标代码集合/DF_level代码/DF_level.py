import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./焦炭连续.csv')

# 计算DF指标
data['df'] = abs(data['open']-data['close']) / abs(data['high'] - data['low'])

'''
下面这段代码看数据分布,绘制数据分布图
plt.hist(data['df'], bins=40)
plt.xlabel('X Label') # 添加 X 轴标签
plt.ylabel('Y Label') # 添加 Y 轴标签
plt.title('Histogram') # 添加图形标题
plt.show()
'''

# 参数一：计算df的百分位值，当df小于n时，统计未来n个周期的收益率。
quan_q = n
pct = data['df'].quantile(q=quan_q)
