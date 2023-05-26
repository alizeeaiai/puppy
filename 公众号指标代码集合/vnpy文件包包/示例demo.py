import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Optional
from decimal import Decimal
from math import floor, ceil

import numpy as np
import talib

from vnpy.trader.object import BarData, TickData
from vnpy.trader.constant import Exchange, Interval


class BarGenerator:
    def __init__(
            self,
            on_bar: Callable,
            window: int = 0,
            on_window_bar: Callable = None,
            interval: Interval = Interval.MINUTE
    ) -> None:
        self.bar: BarData = None
        self.on_bar: Callable = on_bar
        self.interval: Interval = interval
        self.interval_count: int = 0

        self.hour_bar: BarData = None

        self.window: int = window
        self.window_bar: BarData = None
        self.on_window_bar: Callable = on_window_bar

        self.last_tick: TickData = None

    def update_tick(self, tick: TickData) -> None:

        # 程序运行的第一步，new_minute设置为false
        new_minute: bool = False
        # 如果当前的tick为0，就返回
        if not tick.last_price:
            return
        # 如果上一笔tick有值，但是当前tick的时间戳小于上一笔tick的时间戳，就返回。这说明交易所返回了历史数据。
        if self.last_tick and tick.datetime < self.last_tick.datetime:
            return

        # 程序运行第二步，因为在init函数里self.bar为none值，所以进入到这里，先把new_minute设置为True
        if not self.bar:
            new_minute = True
        # 程序运行第五步，new_minute为False，此时先判断时间，如果时间进入下一分钟，那么重置秒。假设没有进入下一分钟
        elif ((self.bar.datetime.minute != tick.datetime.minute)or
              (self.bar.datetime.hour != tick.datetime.hour)):
            self.bar.datetime = self.bar.datetime.replace(second=0,microsecond=0)

            self.on_bar(self.bar)

            new_minute = True

        # 程序运行的第三步，new_minute设置为True，实例化BarData，此时self.bar就有值了
        if new_minute:
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.MINUTE,
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                close_price=tick.last_price,
                low_price=tick.last_price,
                high_price=tick.last_price,
                open_interest=tick.open_interest
            )
        # 程序运行第6步，此时还是在1分钟内。先对self.bar.high_price赋值，它等于上一个tick的价格，和当前tick的价格的最大值
        else:
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            if tick.high_price > self.last_tick.high_price:
                self.bar.high_price = max(self.bar.high_price, tick.high_price)

            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            if tick.low_price < self.last_tick.low_price:
                self.bar.low_price = min(self.bar.low_price, tick.low_price)

            self.bar.close_price = tick.last_price
            self.bar.open_interest = tick.open_interest
            self.bar.datetime = tick.datetime

        if self.last_tick:
            volume_change: float = tick.volume - self.last_tick.volume
            self.bar.volume += max(volume_change, 0)

            turnover_change: float = tick.turnover - self.last_tick.turnover
            self.bar.turnover += max(turnover_change, 9)
        # 程序运行的第四步，把当前的tick赋值给self.last_tick
        self.last_tick = tick

    def update_bar(self, bar: BarData) -> None:
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bar)
        else:
            self.update_bar_hour_window(bar)

    def update_bar_minute_window(self, bar: BarData) -> None:
        # 第1步，self.window_bar开始为None，这里给self.window_bar赋值
        if not self.window_bar:
            dt: datetime = bar.datetime.replace(second=0,microsecond=0)
            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
        # 第2步，此时window bar已经有值，找到self.window_bar中high和low的大小
        else:
            self.window_bar.high_price = max(self.window_bar.high_price, bar.high_price)
            self.window_bar.low_price = min(self.window_bar.low_price, bar.low_price)

            self.window_bar.close_price = bar.close_price
            self.window_bar.volume += bar.volume
            self.window_bar.turnover += bar.turnover
            self.window_bar.open_interest = bar.open_interest

        if not (bar.datetime.minute + 1) % self.window:
            self.on_window_bar(self.window_bar)
            self.window_bar = None

    def update_bar_hour_window(self, bar: BarData) -> None:
        if not self.hour_bar:
            dt: datetime = bar.datetime.replace(second=0, microsecond=0)
            # 如果self.hour_bar为None
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=dt,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            return