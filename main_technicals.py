from utils import write, process_data
from visuals import *
import os
    
def main():
    path = os.path.dirname(__file__) + "\\"
    stock_data = path + "STOCK_DATA.xlsx"
    
    data = process_data(stock_data, "COUR")
    write(stock_data, data)
    
    # Historical
    #stock(data)
    #SMA(data) # momentum/trend - SMA 50 & 200, MACD
    #RSI(data) # momentum/trend - RSI, stochastic RSI, %K, %D
    #FI(data) # volume - force index
    #BB(data) # volatility - bollinger bands low mid high
    #Fib(data) # momentum/trend - fib retracement + extension
    
    # Future
    #pred_prophet(data, period=365) # facebook's prophet model - additive regression model using components: piecewise growth curve trend, fourier series, dummy variables
    pred_ARCH(data, period=365) # Autoregressive Conditional Heteroskedasticity
    
    
    # sizing - kelly criterion
    # VaR - monte carlo
    
    
    
    
if __name__ == "__main__":
    main()