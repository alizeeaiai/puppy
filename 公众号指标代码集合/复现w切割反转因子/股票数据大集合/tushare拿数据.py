import tushare as ts
import pandas as pd
pro = ts.pro_api('93b1a9a8322691fbac916f6376729a3cadbdde2934a2632f89a2fec9')
data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

stock_list = list(data['ts_code'].values)
for each_stock in stock_list:
    df = pro.daily(ts_code=each_stock, start_date='20190101', end_date='20230420')
    df.to_csv('./{}.csv'.format(each_stock))