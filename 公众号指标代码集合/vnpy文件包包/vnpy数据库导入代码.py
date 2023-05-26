from vnpy.trader.setting import SETTINGS
from vnpy_datamanager.engine import ManagerEngine
from vnpy.trader.constant import Exchange, Interval
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
import tushare.futures.domestic_cons as qh

from datetime import datetime

# step 1 设置setting里的变量
# 设置全局变量，后面代码存储时会去获取这些变量
# 此处也可以设置为你喜欢的数据库 https://www.vnpy.com/docs/cn/database.html#
# 在vnpy.trader.setting里，setting是个字典
SETTINGS["database.name"] = "sqlite"
SETTINGS["database.database"] = "database.db"
SETTINGS["database.host"] = ""
SETTINGS["database.port"] = 0
SETTINGS["database.user"] = ""
SETTINGS["database.password"] = ""

# 设置获取源 https://www.vnpy.com/docs/cn/datafeed.html#
SETTINGS["datafeed.name"] = "tushare"
SETTINGS["datafeed.username"] = "token"
SETTINGS["datafeed.password"] = ("8baccf719621c815bbbf6b91b349" +
                                 "c4f36bbde6ee5d9416421ed3875b")

# step 2 获取所有期货代码
# 调用qh.FUTURE_CODE，返回的是一个字典{'IH': ('CFFEX', '上证50指数', 300),
#                   'IF': ('CFFEX', '沪深300指数', 300),
#                   'IC': ('CFFEX', '中证500指数', 200),}

heyue = qh.FUTURE_CODE


managerEngine = ManagerEngine(MainEngine(), EventEngine())

# step 3 循环获取所有期货数据并存入数据库
for key, value in heyue.items():
    count = managerEngine.download_bar_data(symbol=key,
                                            exchange=Exchange(value[0]),
                                            interval=Interval("d"),
                                            start=datetime(2022, 12, 1),
                                            output=print,
                                            )
