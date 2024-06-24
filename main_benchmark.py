import yfinance as yf
import pandas as pd
import numpy as np
import quantstats as qs
import os
import sys
import datetime as dt
import warnings

"""
Dataframes:

kensho (KCEVP:IND), ^KEV, ^SPKEV, ^KEVP - S&P Kensho Capped EV Index
ev > DRIV - Global X Auto & EV ETF
ev > ^SPX - S&P 500
ev > NDAQ - NASDAQ
"""    

# admin
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")
qs.extend_pandas()

def read(f, _t=False, sheet=None): # read csv/xslx: [Date, Open, High, Low, Close, (Change)] 
    if not _t:
        df = pd.read_csv(f, usecols="Date Change".split(), index_col="Date")
    else:
        df = pd.read_excel(f, usecols="Date Change".split(), sheet_name=sheet, index_col="Date") 
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')

    return df

def prep_benchmark(returns, bench, cutoff: list[int] = None): # cutoff: [year, month, day] 
    df = bench.copy()
    earliest = returns.index.min().date()
    latest = returns.index.max().date()
    
    if cutoff is not None:
        latest = dt.date(*cutoff)

    df = df.loc[(pd.to_datetime(df.index).date >= earliest) & (pd.to_datetime(df.index).date <= latest)] # trim

    for i, v in df.items():
        df.loc[i] = returns.loc[i].values[0]
    
    return df, bench

def main(directory, *args, **kwargs):
    filename_kensho, *_ = args
    kensho = read(filename_kensho)
    ndaq = read(dir_ + "indices.xlsx", 1, "Sheet1")
    spy = qs.utils.download_returns("^SPX")

    ken, spy = prep_benchmark(kensho, spy)
    byd = qs.utils.download_returns("BYDDF")
    geely = qs.utils.download_returns("GELYF")

    qs.reports.html(geely, spy, output="C:\\Users\\franc\\Downloads\\qs_report.html")
    
if __name__ == "__main__":
    dir_ = os.path.dirname(__file__) + "\\"
    file_kensho = dir_ + "stockdata_week2.csv"
    
    main(dir_, file_kensho)