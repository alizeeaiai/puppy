import time
import pandas as pd
from scipy.signal import savgol_filter

df = pd.read_csv('./乙二醇连续.csv')
# 程序开始
start_time = time.time()

df['MA'] = df['close'].rolling(21).mean()
# 设置参数窗口大小为21，阶数为5
df['sav_gol'] = savgol_filter(df['close'], window_length = 21, polyorder = 5)

# 程序结束
end_time = time.time()

# 计算程序用时，单位为秒
elapsed_time = end_time - start_time
print("程序用时:", elapsed_time, "秒")