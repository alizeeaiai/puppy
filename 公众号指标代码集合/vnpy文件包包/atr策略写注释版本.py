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


class AtrRsiStrategy(CtaTemplate):
    """"""

    author = "用Python的交易员"

    atr_length = 22
    atr_ma_length = 10
    rsi_length = 5
    rsi_entry = 16
    trailing_percent = 0.8
    fixed_size = 1

    atr_value = 0
    atr_ma = 0
    rsi_value = 0
    rsi_buy = 0
    rsi_sell = 0
    intra_trade_high = 0
    intra_trade_low = 0

    parameters = [
        "atr_length",
        "atr_ma_length",
        "rsi_length",
        "rsi_entry",
        "trailing_percent",
        "fixed_size"
    ]
    variables = [
        "atr_value",
        "atr_ma",
        "rsi_value",
        "rsi_buy",
        "rsi_sell",
        "intra_trade_high",
        "intra_trade_low"
    ]

    # step 1 初始化函数
    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        # 初始化父类
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        # 生成bar
        self.bg = BarGenerator(self.on_bar)
        # 生成bar池子
        self.am = ArrayManager()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

        self.rsi_buy = 50 + self.rsi_entry
        self.rsi_sell = 50 - self.rsi_entry

        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        # am是arraymanager，调用了atr函数
        atr_array = am.atr(self.atr_length, array=True)
        self.atr_value = atr_array[-1]
        self.atr_ma = atr_array[-self.atr_ma_length:].mean()
        self.rsi_value = am.rsi(self.rsi_length)

        # 如果仓位为0
        if self.pos == 0:
            self.intra_trade_high = bar.high_price
            self.intra_trade_low = bar.low_price

            if self.atr_value > self.atr_ma:
                if self.rsi_value > self.rsi_buy:
                    # 如果rsi_value大于rsi_buy，就以收盘价+5个价位买入固定数量的
                    self.buy(bar.close_price + 5, self.fixed_size)
                elif self.rsi_value < self.rsi_sell:
                    # 如果rsi_value小于rsi_sell，就以收盘价-5个价格卖出固定数量的
                    self.short(bar.close_price - 5, self.fixed_size)

        # 如果pos大于0，也就是有多头仓位，此时用sell函数平仓
        elif self.pos > 0:
            # 给self.intra_trade_high赋值，取开仓时self.intra_trade_high和当前的bar.high_price
            self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
            self.intra_trade_low = bar.low_price

            long_stop = self.intra_trade_high * \
                (1 - self.trailing_percent / 100)
            self.sell(long_stop, abs(self.pos), stop=True)

        # 当pos小于0，也就是有空头仓位，用cover平仓
        elif self.pos < 0:
            self.intra_trade_low = min(self.intra_trade_low, bar.low_price)
            self.intra_trade_high = bar.high_price

            short_stop = self.intra_trade_low * \
                (1 + self.trailing_percent / 100)
            self.cover(short_stop, abs(self.pos), stop=True)

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

'''
第1步，在初始化__init__函数里，传入4个参数，分别是：cta_engine、strategy_name、vt_symbol和setting。
在初始化函数里，同时初始化它的父类，也就是cta_template。
在初始化函数里，实例化2个类。一个是BarGenerator，另一个是ArrayManager。
在Barenerator中传入self.on_bar函数
'''