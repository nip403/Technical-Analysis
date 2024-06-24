from utils import write, process_data
from visuals import *
import os
    
def main():
    path = os.path.dirname(__file__) + "\\"
    stock_data = path + "STOCK_DATA.xlsx"
    
    cour = process_data(stock_data, "COUR")
    write(stock_data, cour)
    
    # visualise
    
    stock(cour, plot_volume=0, candlestick=0)
    SMA(cour, candlestick=True) # 50/200-day SMA, MACD (100-25 windows) (altered ta module source wrapper.py)
    BB(cour, candlestick=True) # 200 period bollinger bands
    RSI(cour) # RSI, Stochastic RSI, %K, %D - 50 day window
    
if __name__ == "__main__":
    main()