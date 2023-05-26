import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 创建画布
fig = plt.figure(figsize=(8, 6), dpi=80)
# 添加子图
ax = fig.add_subplot(1, 1, 1)

df = pd.read_csv('./LPG连续.csv')

# 对数据进行归一化处理
y1_norm = MinMaxScaler().fit_transform((df['close'][-100:]).to_numpy().reshape(-1, 1)).flatten()
y2_norm = MinMaxScaler().fit_transform(df['er'][-100:].to_numpy().reshape(-1, 1)).flatten()

# 绘制折线图1
ax.plot(df['trade_date'][-100:], y1_norm, color='blue', linewidth=1, label='close_price')
# 绘制折线图2
ax.plot(df['trade_date'][-100:], y2_norm, color='red', linewidth=1, linestyle='--', label='ER')

# 添加图例
ax.legend()

# 添加标题和坐标轴标签
ax.set_title('close_price and ER', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)

# 显示图像
plt.show()