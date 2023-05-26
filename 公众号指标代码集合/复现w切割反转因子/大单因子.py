from typing import Any
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

class DemoStrategy(CtaTemplate):

    author = 'Puppy'

    fast_window = 10
    slow_window = 20

    fast_ma0 = 0.0
    fast_ma1 = 0.0
    slow_ma0 = 0.0
    slow_ma1 = 0.0

    parameters = ['fast_window', 'slow_window']

    variables = ['fast_ma0',
                 'fast_ma1',
                 'slow_ma0',
                 'slow_ma1']



    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_init(self):
        '''策略初始化'''
        self.write_log('策略初始化')
        '''调用10天的数据'''
        self.load_bar(10)

    def on_start(self):
        '''策略启动'''
        self.write_log('策略启动')

    def on_stop(self):
        '''策略停止'''
        self.write_log('策略停止')

    def on_tick(self, tick: TickData):
        '''tick更新'''
        self.bg.update_tick(tick) # 合成1分钟k线

    def on_bar(self, bar: BarData):
        '''k线更新'''
        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
        # 计算技术指标
        fast_ma = am.sma(self.fast_window, array=True)
        self.fast_ma0 = fast_ma[-1]
        self.fast_ma1 = fast_ma[-2]

        slow_ma = am.sma(self.slow_window, array=True)
        self.slow_ma0 = slow_ma[-1]
        self.slow_ma1 = slow_ma[-2]

        # 判断金叉死叉
        cross_over = (self.fast_ma0 > self.slow_ma0 and self.fast_ma1 < self.slow_ma1)
        cross_below = (self.fast_ma0 < self.slow_ma0 and self.fast_ma1 > self.slow_ma1)

        if cross_over:
            price = bar.close_price + 5

            if not self.pos:
                self.buy(price ,1)
            elif self.pos < 0:
                self.cover(price ,1)
                self.buy(price, 1)
        elif cross_below:
            price = bar.close_price - 5
            if not self.pos:
                self.short(price, 1)
            elif self.pos > 0:
                self.sell(price, 1)
                self.buy(price, 1)

        self.put_even()


if __name__ == '__main__':
    cta_engine = 1
    strategy_name = "try_try_try"
    vt_symbol = 'rb888'
    setting = {'pig': 'dog'}
    strategy = DemoStrategy(cta_engine, strategy_name, vt_symbol, setting)

