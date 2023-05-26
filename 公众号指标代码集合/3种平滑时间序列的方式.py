import pandas as pd
from scipy.signal import savgol_filter
df = pd.DataFrame()

# 第一部分：局部多项式拟合曲线

# 假设df是一个包含价格数据的DataFrame，第一个参数窗口大小设置为21，第二个参数阶数设置为5。

df['MA'] = df['Close'].rolling(21).mean()
df['sav_gol'] = savgol_filter(df['Close'], window_length = 21, polyorder = 5)

# 绘制图像
df.plot(y=['Close', 'MA', 'sav_gol'])

import numpy as np
import numba as nb
import pandas as pd


# 第二部分：热方程平滑时间序列



@nb.jit(nopython = True)  # 用numba库，将Python代码转换为机器码，运行更快
# 定义函数，传入2个参数。prices是numpy数组，t_end是时间长度
def explicit_heat_smooth(prices: np.array, t_end: float = 3.0) -> np.array:

    k = 0.1  # 定义时间步长k，即在每个时间步长内迭代一次，使得平滑足够充分

    P = prices  # 设置初始条件，即将初始价格向量赋值给P

    t = 0
    while t < t_end:
        # 使用热传导方程式求解，实现平滑的核心步骤。采用了差分方案，将相邻的价格差平均分配给前后的价格，使得价格趋势更加平滑
        P = k * (P[2:] + P[:-2]) + P[1:-1] * (1 - 2 * k)

        # 将边界条件添加到处理后的价格向量P的两端
        P = np.hstack((
            np.array([prices[0]]),
            P,
            np.array([prices[-1]]),
        ))
        t += k  # 更新时间变量t，进入下一个时间步长

    return P
