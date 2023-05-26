import numpy as np
import pandas as pd
import talib as ta
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType

def obvm():
    '''计算obvm和signal obvm'''
    df = pd.read_excel('./510050_15.xlsx')
    df['obv'] = ta.OBV(df['close'], df['volume']) / 1e10
    df['obvm'] = ta.EMA(df['obv'], timeperiod=7)
    df['signal_obvm'] = ta.EMA(df['obv'], timeperiod=50)
    df = df.replace([np.nan], 0.0)
    return df

def draw(df):
    '''绘制均线和k线'''
    kline = Kline(init_opts=opts.InitOpts(width='1200px',
                      height='600px', theme=ThemeType.DARK))
    kline.add_xaxis(df.index.tolist()) # 设置时间轴
    y = list(df.loc[:, ['open', 'high', 'low', 'close']].round(
            4).values)
    y = [i.tolist() for i in y]
    kline.add_yaxis('Kline', y)
    kline.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    kline.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                is_scale=True, axislabel_opts=opts.LabelOpts(rotate=60)),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}")),
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            title_opts=opts.TitleOpts(
                title="510050_50ETF_Kline_and_MA", pos_left='45%'),  # 题目位置
            legend_opts=opts.LegendOpts(pos_right="35%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )
    line1 = Line()
    line1.add_xaxis(df.index.tolist())
    line1.add_yaxis('obvm', df.obvm.tolist(), is_smooth=True)
    line1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
    line1.set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
            legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
            tooltip_opts=opts.TooltipOpts(
                trigger="axis", axis_pointer_type="cross")  # 添加趋势线
        )
    kline.overlap(line1)

    line2 = Line()
    line2.add_xaxis(df.index.tolist())
    line2.add_yaxis('obvm_signal', df.signal_obvm.tolist(), is_smooth=True)
    line2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 是否显示数据标签
    line2.set_global_opts(
        datazoom_opts=[opts.DataZoomOpts(type_='inside')],  # 内部滑动
        legend_opts=opts.LegendOpts(pos_right="20%", pos_top="5%"),  # 图例位置
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross")  # 添加趋势线
    )
    line1.overlap(line2)
    kline.overlap(line2)
    kline.render('./obvm_painting.html')
    return line1


if __name__ == '__main__':
    df = obvm()
    print(df.obvm)
    print(df.signal_obvm)
    draw(df)