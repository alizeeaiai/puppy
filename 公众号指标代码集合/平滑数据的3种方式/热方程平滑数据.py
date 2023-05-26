import time
import pandas as pd
import numba as nb
import numpy as np

df = pd.read_csv('./乙二醇连续.csv')
# 获取程序开始时间
start_time = time.time()


@nb.jit(forceobj=True)
def heat_smooth():
    t = 0
    t_end = 812
    k = 0.1  # 定义时间步长k，即在每个时间步长内迭代一次，使得平滑足够充分
    price_array = df.close.to_numpy()
    P = price_array  # 设置初始条件，即将初始价格向量赋值给P
    while t < t_end:
        # 使用热传导方程式求解，实现平滑的核心步骤。采用了差分方案，将相邻的价格差平均分配给前后的价格，使得价格趋势更加平滑
        P = k * (P[2:] + P[:-2]) + P[1:-1] * (1 - 2 * k)

        # 将边界条件添加到处理后的价格向量P的两端
        P = np.hstack((
            np.array([price_array[0]]),
            P,
            np.array([price_array[-1]]),
            ))
        t += k  # 更新时间变量t，进入下一个时间步长
    # 获取程序结束时间
heat_smooth()
end_time = time.time()

# 计算程序用时，单位为秒
elapsed_time = end_time - start_time
print("程序用时:", elapsed_time, "秒")
