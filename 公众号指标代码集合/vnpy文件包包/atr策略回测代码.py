from vnpy_ctastrategy import backtesting
from vnpy_ctastrategy.strategies.atr_rsi_strategy import AtrRsiStrategy
from datetime import datetime

# step 1 实例化backtesting
engine = backtesting.BacktestingEngine()

# step 2 传入参数
engine.set_parameters(
    vt_symbol='MA888.CZCE',
    interval="1m",
    start=datetime(2023,1,1),
    end=datetime(2023,5,1),
    rate=0.003,
    slippage=0.01,
    size=100,
    pricetick=1,
    capital=100
)

# step 3 调用engine中的add_strategy函数
engine.add_strategy(AtrRsiStrategy, {})

# step 4
engine.load_data()
# step 5 启动回测引擎
engine.run_backtesting()
# step 6 计算结果
df = engine.calculate_result()
# step 7 计算数据
engine.calculate_statistics()
engine.show_chart()
