import tushare as ts
import pandas as pd

pro = ts.pro_api('8baccf719621c815bbbf6b91b349c4f36bbde6ee5d9416421ed3875b')
df = pro.fut_basic(exchange='CZCE', fut_type='2', fields='ts_code,symbol,name,list_date,delist_date')
df.to_csv('./第二个文本.csv')