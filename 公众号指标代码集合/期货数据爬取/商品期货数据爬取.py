import datetime
import warnings
import time
from typing import Dict
from typing import List
import re
import pandas as pd
from akshare.futures import cons
import requests
import random
import os
from fake_useragent import UserAgent
warnings.filterwarnings('ignore')

calendar = cons.get_calendar()
DATE_PATTERN = re.compile(r"^([0-9]{4})[-/]?([0-9]{2})[-/]?([0-9]{2})")


def convert_date(date):
    """
    transform a date string to datetime.date object
    :param date, string, e.g. 2016-01-01, 20160101 or 2016/01/01
    :return: object of datetime.date(such as 2016-01-01) or None
    """
    if isinstance(date, datetime.date):
        return date
    elif isinstance(date, str):
        match = DATE_PATTERN.match(date)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                return datetime.date(
                    year=int(groups[0]),
                    month=int(groups[1]),
                    day=int(groups[2]),
                )
    return None


def random_sleep(mu=1, sigma=0.4):
    '''正态分布随机睡眠
    :param mu: 平均值
    :param sigma: 标准差，决定波动范围
    '''
    secs = random.normalvariate(mu, sigma)
    if secs <= 0:
        secs = mu  # 太小则重置为平均值
    time.sleep(secs)

def chinese_to_english(chinese_var: str):
    """
    映射期货品种中文名称和英文缩写
    :param chinese_var: 期货品种中文名称
    :return: 对应的英文缩写
    """
    chinese_list = [
        "橡胶",
        "天然橡胶",
        "石油沥青",
        "沥青",
        "沥青仓库",
        "沥青(仓库)",
        "沥青厂库",
        "沥青(厂库)",
        "热轧卷板",
        "热轧板卷",
        "燃料油",
        "白银",
        "线材",
        "螺纹钢",
        "铅",
        "铜",
        "铝",
        "锌",
        "黄金",
        "钯金",
        "锡",
        "镍",
        "纸浆",
        "豆一",
        "大豆",
        "豆二",
        "胶合板",
        "玉米",
        "玉米淀粉",
        "聚乙烯",
        "LLDPE",
        "LDPE",
        "豆粕",
        "豆油",
        "大豆油",
        "棕榈油",
        "纤维板",
        "鸡蛋",
        "聚氯乙烯",
        "PVC",
        "聚丙烯",
        "PP",
        "焦炭",
        "焦煤",
        "铁矿石",
        "乙二醇",
        "强麦",
        "强筋小麦",
        " 强筋小麦",
        "硬冬白麦",
        "普麦",
        "硬白小麦",
        "硬白小麦（）",
        "皮棉",
        "棉花",
        "一号棉",
        "白糖",
        "PTA",
        "菜籽油",
        "菜油",
        "早籼稻",
        "早籼",
        "甲醇",
        "柴油",
        "玻璃",
        "油菜籽",
        "菜籽",
        "菜籽粕",
        "菜粕",
        "动力煤",
        "粳稻",
        "晚籼稻",
        "晚籼",
        "硅铁",
        "锰硅",
        "硬麦",
        "棉纱",
        "苹果",
        "原油",
        "中质含硫原油",
        "尿素",
        "20号胶",
        "苯乙烯",
        "不锈钢",
        "粳米",
        "20号胶20",
        "红枣",
        "不锈钢仓库",
        "纯碱",
        "液化石油气",
        "低硫燃料油",
        "纸浆仓库",
        "石油沥青厂库",
        "石油沥青仓库",
        "螺纹钢仓库",
        "螺纹钢厂库",
        "纸浆厂库",
        "低硫燃料油仓库",
        "低硫燃料油厂库",
        "短纤",
        '涤纶短纤',
        '生猪',
        '花生',
    ]
    english_list = [
        "RU",
        "RU",
        "BU",
        "BU",
        "BU",
        "BU",
        "BU2",
        "BU2",
        "HC",
        "HC",
        "FU",
        "AG",
        "WR",
        "RB",
        "PB",
        "CU",
        "AL",
        "ZN",
        "AU",
        "AU",
        "SN",
        "NI",
        "SP",
        "A",
        "A",
        "B",
        "BB",
        "C",
        "CS",
        "L",
        "L",
        "L",
        "M",
        "Y",
        "Y",
        "P",
        "FB",
        "JD",
        "V",
        "V",
        "PP",
        "PP",
        "J",
        "JM",
        "I",
        "EG",
        "WH",
        "WH",
        "WH",
        "PM",
        "PM",
        "PM",
        "PM",
        "CF",
        "CF",
        "CF",
        "SR",
        "TA",
        "OI",
        "OI",
        "RI",
        "ER",
        "MA",
        "MA",
        "FG",
        "RS",
        "RS",
        "RM",
        "RM",
        "ZC",
        "JR",
        "LR",
        "LR",
        "SF",
        "SM",
        "WT",
        "CY",
        "AP",
        "SC",
        "SC",
        "UR",
        "NR",
        "EB",
        "SS",
        "RR",
        "NR",
        "CJ",
        "SS",
        "SA",
        "PG",
        "LU",
        "SP",
        "BU",
        "BU",
        "RB",
        "RB",
        "SP",
        "LU",
        "LU",
        "PF",
        "PF",
        "LH",
        "PK",
    ]
    pos = chinese_list.index(chinese_var)
    return english_list[pos]


def pandas_read_html_link(url: str, encoding: str = "utf-8", method: str = "get", data: Dict = None,
                          headers: Dict = None):
    """
    利用 pandas 提供的 read_html 函数来直接提取网页中的表格内容, 如网站链接失败, 可重复爬取 20 次
    :param url: string 网站地址
    :param encoding: string 编码类型: "utf-8", "gbk", "gb2312"
    :param method: string 访问方法: "get", "post"
    :param data: dict 上传数据: 键值对
    :param headers: dict 游览器请求头: 键值对
    :return: requests.response 爬取返回内容: response
    """
    i = 0
    while True:
        try:
            headers = {'User-Agent': str(UserAgent().random)}
            if method == "get":
                r = requests.get(url, headers=headers, timeout=20)
                r.encoding = encoding
                r = pd.read_html(r.text, encoding=encoding)
                return r
            elif method == "post":
                r = requests.post(url, timeout=20, data=data, headers=headers)
                r.encoding = encoding
                r = pd.read_html(r.text, encoding=encoding)
                return r
            else:
                raise ValueError("请提供正确的请求方式")
        except requests.exceptions.Timeout as e:
            i += 1
            print(f"第{str(i)}次链接失败, 最多尝试20次", e)
            time.sleep(5)
            if i > 20:
                return None


def _join_head(content: pd.DataFrame) -> List:
    headers = []
    for s1, s2 in zip(content.iloc[0], content.iloc[1]):
        if s1 != s2:
            s = f'{s1}{s2}'
        else:
            s = s1
        headers.append(s)
    return headers


def _check_information(df_data, date):
    """
    数据验证和计算模块
    :param df_data: pandas.DataFrame 采集的数据
    :param date: datetime.date 具体某一天 YYYYMMDD
    :return: pandas.DataFrame
    中间数据
    symbol  spot_price near_contract  ...  near_basis_rate dom_basis_rate      date
     CU    49620.00        cu1811  ...        -0.002418      -0.003426  20181108
     RB     4551.54        rb1811  ...        -0.013521      -0.134359  20181108
     ZN    22420.00        zn1811  ...        -0.032114      -0.076271  20181108
     AL    13900.00        al1812  ...         0.005396       0.003957  20181108
     AU      274.10        au1811  ...         0.005655       0.020430  20181108
     WR     4806.25        wr1903  ...        -0.180026      -0.237035  20181108
     RU    10438.89        ru1811  ...        -0.020969       0.084406  20181108
     PB    18600.00        pb1811  ...        -0.001344      -0.010215  20181108
     AG     3542.67        ag1811  ...        -0.000754       0.009408  20181108
     BU     4045.53        bu1811  ...        -0.129904      -0.149679  20181108
     HC     4043.33        hc1811  ...        -0.035449      -0.088128  20...
    """
    df_data = df_data.loc[:, [0, 1, 2, 3, 5, 6]]
    df_data.columns = [
        "symbol",
        "spot_price",
        "near_contract",
        "near_contract_price",
        "dominant_contract",
        "dominant_contract_price",
    ]
    records = pd.DataFrame()
    for string in df_data["symbol"].tolist():
        if string == "PTA":
            news = "PTA"
        else:
            news = "".join(re.findall(r"[\u4e00-\u9fa5]", string))
        if news != "" and news not in ["商品", "价格", "上海期货交易所", "郑州商品交易所", "大连商品交易所"]:
            symbol = chinese_to_english(news)
            record = pd.DataFrame(df_data[df_data["symbol"] == string])
            record.loc[:, "symbol"] = symbol
            record.loc[:, "spot_price"] = record.loc[:, "spot_price"].astype(float)
            if (
                    symbol == "JD"
            ):  # 鸡蛋现货为元/公斤, 鸡蛋期货为元/500千克, 其余元/吨(http://www.100ppi.com/sf/)
                record.loc[:, "spot_price"] = float(record["spot_price"]) * 500
            elif (
                    symbol == "FG"
            ):  # 上表中现货单位为元/平方米, 期货单位为元/吨. 换算公式：元/平方米*80=元/吨(http://www.100ppi.com/sf/959.html)
                record.loc[:, "spot_price"] = float(record["spot_price"]) * 80
            records = records._append(record)

    records.loc[
    :, ["near_contract_price", "dominant_contract_price", "spot_price"]
    ] = records.loc[
        :, ["near_contract_price", "dominant_contract_price", "spot_price"]
        ].astype(
        "float"
    )

    records.loc[:, "near_contract"] = records["near_contract"].replace(
        r"[^0-9]*(\d*)$", r"\g<1>", regex=True
    )
    records.loc[:, "dominant_contract"] = records["dominant_contract"].replace(
        r"[^0-9]*(\d*)$", r"\g<1>", regex=True
    )

    records.loc[:, "near_contract"] = records["symbol"] + records.loc[
                                                          :, "near_contract"
                                                          ].astype("int").astype("str")
    records.loc[:, "dominant_contract"] = records["symbol"] + records.loc[
                                                              :, "dominant_contract"
                                                              ].astype("int").astype("str")

    records["near_contract"] = records["near_contract"].apply(
        lambda x: x.lower()
        if x[:-4]
           in cons.market_exchange_symbols["shfe"] + cons.market_exchange_symbols["dce"]
        else x
    )
    records.loc[:, "dominant_contract"] = records.loc[:, "dominant_contract"].apply(
        lambda x: x.lower()
        if x[:-4]
           in cons.market_exchange_symbols["shfe"] + cons.market_exchange_symbols["dce"]
        else x
    )
    records.loc[:, "near_contract"] = records.loc[:, "near_contract"].apply(
        lambda x: x[:-4] + x[-3:]
        if x[:-4] in cons.market_exchange_symbols["czce"]
        else x
    )
    records.loc[:, "dominant_contract"] = records.loc[:, "dominant_contract"].apply(
        lambda x: x[:-4] + x[-3:]
        if x[:-4] in cons.market_exchange_symbols["czce"]
        else x
    )

    records["near_basis"] = records["near_contract_price"] - records["spot_price"]
    records["dom_basis"] = records["dominant_contract_price"] - records["spot_price"]
    records["near_basis_rate"] = (
            records["near_contract_price"] / records["spot_price"] - 1
    )
    records["dom_basis_rate"] = (
            records["dominant_contract_price"] / records["spot_price"] - 1
    )
    records.loc[:, "date"] = date.strftime("%Y%m%d")
    return records


def futures_spot_price_previous(date: str = "20220209") -> pd.DataFrame:
    """
    具体交易日大宗商品现货价格及相应基差表
    http://www.100ppi.com/sf2/day-2017-09-12.html
    :param date: 交易日; 历史日期
    :type date: str
    :return: 现货价格及相应基差
    :rtype: pandas.DataFrame

           商品     现货价格 主力合约代码  主力合约价格   主力合约基差 主力合约变动百分比 180日内主力基差最高 180日内主力基差最低 180日内主力基差平均
        0       铜  70491.7   2203   70240   251.00      0.36     3020.00     -335.00      534.19
        1     螺纹钢  4787.78   2205    4881   -93.00     -1.94      839.00     -272.56      234.91
        2       锌    25196   2203   25180    16.00      0.06      865.00     -142.00      507.50
        3       铝    22900   2203   22805    95.00      0.41      243.33     -430.00      -71.12
        4      黄金   373.66   2206  376.02    -2.36     -0.63        3.49       -4.50       -1.06
        5      线材     4988   2205    5186  -198.00     -3.97      846.00     -699.00      140.27
        6     燃料油     5560   2205    3195  2365.00     42.54     3228.00     1867.00     2362.36
        7    天然橡胶    13772   2205   14610  -838.00     -6.08     -765.00    -1585.00    -1079.41
        8       铅    14950   2203   14915    35.00      0.23      235.00     -203.33       22.53
        9      白银     4791   2206    4812   -21.00     -0.44       56.00     -195.67      -50.76
        10   石油沥青   3701.2   2206    3656    45.00      1.22      668.00     -174.00      186.64
        11   热轧卷板     5006   2205    5033   -27.00     -0.54      466.00      -35.00      222.85
        12      镍   173033   2203  170090  2943.00      1.70     5500.00     -350.00     2232.43
        13      锡   333912   2203  331010  2902.00      0.87    16770.00     1970.00     8322.93
        14     纸浆     6360   2205    6368    -8.00     -0.13      688.00     -350.00      122.90
        15    不锈钢  18306.7   2203   18140   166.00      0.91     2708.33     -778.33     1116.30
        16    PTA     5675   2205    5714   -39.00     -0.69       92.73     -237.00      -32.29
        17     白糖     5690   2205    5716   -26.00     -0.46       32.00     -329.00     -158.40
        18     棉花  22904.2   2205   21780  1124.00      4.91     2851.00    -1068.33     1102.48
        19     普麦     2870   2203    2421   449.00     15.64      449.00     -365.00       27.90
        20  菜籽油OI    12886   2205   12098   788.00      6.12     1001.00      -74.50      351.44
        21   强麦WH     2870   2205    2931   -61.00     -2.13      449.00     -365.00       27.90
        22     玻璃    25.86   2205    2284  -215.00    -10.39      812.00     -264.20      318.18
        23    菜籽粕     3298   2205    3391   -93.00     -2.82      368.33     -132.00      144.27
        24    油菜籽     6285   2208    6002   283.00      4.50      584.00     -501.00       25.00
        25     硅铁   8587.5   2205    9146  -558.00     -6.50     3192.00    -1491.33      -72.42
        26     锰硅     8150   2205    8326  -176.00     -2.16     3720.00    -2268.00      164.58
        27   甲醇MA     2745   2205    2861  -116.00     -4.23      753.00     -348.00       16.65
        28  动力煤ZC     1160   2205   866.2   293.00     25.26      981.60       83.00      328.78
        29     棉纱    30800   2205   29050  1750.00      5.68     4483.33    -1018.33     2066.71
        30     尿素     2690   2205    2648    42.00      1.56      562.00     -237.33      159.02
        31     纯碱     2700   2205    2948  -248.00     -9.19     1210.00     -429.33      362.94
        32   涤纶短纤     7822   2205    7798    24.00      0.31      951.67     -286.00      224.10
        33    棕榈油    11230   2205    9814  1416.00     12.61     1709.00      420.00      997.82
        34   聚氯乙烯     8700   2205    9173  -473.00     -5.44     3020.00     -620.00      415.66
        35    聚乙烯     9260   2205    9140   120.00      1.30      604.00     -485.00       96.78
        36     豆一     6052   2203    6294  -242.00     -4.00      312.00     -513.00      -94.37
        37     豆粕     4078   2205    3725   353.00      8.66      521.00      108.00      345.46
        38     豆油    10458   2205    9794   664.00      6.35     1342.00       64.00      703.65
        39     玉米  2668.57   2205    2777  -108.00     -4.05      276.71     -110.00       52.48
        40     焦炭     2794   2205    3087  -293.00    -10.49     1129.50     -635.50       95.19
        41     焦煤     2665   2205    2402   263.00      9.87     1577.33      -44.83      584.89
        42    铁矿石   943.33   2205     796   147.00     15.58      316.72       53.28      150.34
        43     鸡蛋     8.02   2205    4244  -234.00     -5.84      932.00     -234.00      514.46
        44    聚丙烯     8700   2205    8660    40.00      0.46      306.00     -630.67      -18.30
        45   玉米淀粉     3256   2203    3128   128.00      3.93      614.67       75.00      371.69
        46    乙二醇  5383.33   2205    5334    49.00      0.91      539.33     -590.00      125.67
        47    苯乙烯     9290   2203    9014   276.00      2.97      591.00     -415.00      145.17
        48  液化石油气     5904   2203    5058   846.00     14.33     1143.00     -925.00      291.68
        49     生猪    13.77   2205   14735  -965.00     -7.01     3925.00    -4050.00     -119.04
    """
    date = convert_date(date) if date is not None else datetime.date.today()
    if date < datetime.date(2011, 1, 4):
        raise Exception("数据源开始日期为 20110104, 请将获取数据时间点设置在 20110104 后")
    if date.strftime("%Y%m%d") not in calendar:
        warnings.warn(f"{date.strftime('%Y%m%d')}非交易日")
        return
    url = date.strftime('http://www.100ppi.com/sf2/day-%Y-%m-%d.html')
    content = pandas_read_html_link(url)
    main = content[1]
    # Header
    header = _join_head(main)
    # Values
    values = main[main[4].str.endswith('%')]
    values.columns = header
    # Basis
    basis = pd.concat(content[2:-1])
    basis.columns = ['主力合约基差', '主力合约基差(%)']
    basis['商品'] = values['商品'].tolist()
    basis = pd.merge(values[["商品", "现货价格", "主力合约代码", "主力合约价格"]], basis)
    basis = pd.merge(basis, values[["商品", "180日内主力基差最高", "180日内主力基差最低", "180日内主力基差平均"]])
    basis.columns = [
        "商品",
        "现货价格",
        "主力合约代码",
        "主力合约价格",
        "主力合约基差",
        "主力合约变动百分比",
        "180日内主力基差最高",
        "180日内主力基差最低",
        "180日内主力基差平均",
    ]
    basis['主力合约变动百分比'] = basis['主力合约变动百分比'].str.strip("%")
    return basis


def futures_spot_price(date: str = "20210201", vars_list: list = cons.contract_symbols) -> pd.DataFrame:
    """
    指定交易日大宗商品现货价格及相应基差
    http://www.100ppi.com/sf/day-2017-09-12.html
    :param date: 开始日期 format: YYYY-MM-DD 或 YYYYMMDD 或 datetime.date 对象; 为空时为当天
    :param vars_list: 合约品种如 RB、AL 等列表 为空时为所有商品
    :return: pandas.DataFrame
    展期收益率数据:
    var              商品品种                     string
    sp               现货价格                     float
    near_symbol      临近交割合约                  string
    near_price       临近交割合约结算价             float
    dom_symbol       主力合约                     string
    dom_price        主力合约结算价                float
    near_basis       临近交割合约相对现货的基差      float
    dom_basis        主力合约相对现货的基差         float
    near_basis_rate  临近交割合约相对现货的基差率    float
    dom_basis_rate   主力合约相对现货的基差率       float
    date             日期                         string YYYYMMDD
    """
    date = cons.convert_date(date) if date is not None else datetime.date.today()
    if date < datetime.date(2011, 1, 4):
        raise Exception("数据源开始日期为 20110104, 请将获取数据时间点设置在 20110104 后")
    if date.strftime("%Y%m%d") not in calendar:
        warnings.warn(f"{date.strftime('%Y%m%d')}非交易日")
        return
    u1 = cons.SYS_SPOT_PRICE_LATEST_URL
    u2 = cons.SYS_SPOT_PRICE_URL.format(date.strftime("%Y-%m-%d"))
    i = 1
    while True:
        for url in [u2, u1]:
            try:
                # url = u2
                r = pandas_read_html_link(url)
                string = r[0].loc[1, 1]
                news = "".join(re.findall(r"[0-9]", string))
                if news[3:11] == date.strftime("%Y%m%d"):
                    records = _check_information(r[1], date)
                    records.index = records["symbol"]
                    var_list_in_market = [i for i in vars_list if i in records.index]
                    temp_df = records.loc[var_list_in_market, :]
                    temp_df.reset_index(drop=True, inplace=True)
                    return temp_df
                else:
                    time.sleep(3)
            except:
                print(f"{date.strftime('%Y-%m-%d')}日生意社数据连接失败，第{str(i)}次尝试，最多5次")
                i += 1
                if i > 5:
                    print(
                        f"{date.strftime('%Y-%m-%d')}日生意社数据连接失败, 如果当前交易日是 2018-09-12, 由于生意社源数据缺失, 无法访问, 否则为重复访问已超过5次，您的地址被网站墙了，请保存好返回数据，稍后从该日期起重试"
                    )
                    return False


def futures_spot_price_daily(
        start_day: str = "20210201",
        end_day: str = "20210208",
        vars_list=cons.contract_symbols,
):
    """
    指定时间段内大宗商品现货价格及相应基差
    http://www.100ppi.com/sf/
    :param start_day: str 开始日期 format:YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象; 默认为当天
    :param end_day: str 结束数据 format:YYYY-MM-DD 或 YYYYMMDD 或 datetime.date对象; 默认为当天
    :param vars_list: list 合约品种如 [RB, AL]; 默认参数为所有商品
    :return: pandas.DataFrame
    展期收益率数据:
    var               商品品种                      string
    sp                现货价格                      float
    near_symbol       临近交割合约                  string
    near_price        临近交割合约结算价             float
    dom_symbol        主力合约                      string
    dom_price         主力合约结算价                 float
    near_basis        临近交割合约相对现货的基差      float
    dom_basis         主力合约相对现货的基差          float
    near_basis_rate   临近交割合约相对现货的基差率    float
    dom_basis_rate    主力合约相对现货的基差率        float
    date              日期                          string YYYYMMDD
    """
    start_day = (
        convert_date(start_day) if start_day is not None else datetime.date.today()
    )
    end_day = (
        convert_date(end_day)
        if end_day is not None
        else convert_date(cons.get_latest_data_date(datetime.datetime.now()))
    )
    df_list = []
    while start_day <= end_day:
        temp_df = futures_spot_price(start_day, vars_list)
        if temp_df is False:
            return pd.concat(df_list).reset_index(drop=True)
        elif temp_df is not None:
            df_list.append(temp_df)
        random_sleep()
        start_day += datetime.timedelta(days=1)
    if len(df_list) > 0:
        temp_df = pd.concat(df_list)
        temp_df.reset_index(drop=True, inplace=True)
        return temp_df


if __name__ == "__main__":
    futures_spot_price_daily_df = futures_spot_price_daily(
        start_day="20190101", end_day="20191230"
    )
    print(futures_spot_price_daily_df)
    futures_spot_price_daily_df.to_csv('futures_spot_price20190101-20191230.csv')

