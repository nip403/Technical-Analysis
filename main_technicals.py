from utils import write, process_data, process_data_ticker
from core import VaR, exit_strategy
from visuals import *
import os
    
def main():
    path = os.path.dirname(__file__) + "\\"
    stock_data = path + "STOCK_DATA.xlsx"
    
    data = process_data(stock_data, "IAU")
    data = data.tail(1000) # last ~2 yrs of trading
    write(stock_data, data)
    
    # Historical
    """
    stock(data)
    SMA(data) # momentum/trend - SMA 50 & 200, MACD
    RSI(data) # momentum/trend - RSI, stochastic RSI, %K, %D
    FI(data) # volume - force index
    BB(data) # volatility - bollinger bands low mid high
    Fib(data) # momentum/trend - fib retracement + extension
    """
    #BB(data)
    
    #Fib(data)
    print(exit_strategy(
        stop_loss=38.39,
        price=43.46,
        pct_portfolio=0.25,
        pct_capital_to_risk=None,
    ))
    
    #all_visual(data)
    
    # Future
    #"""
    #pred_prophet(data, period=365) # facebook's prophet model - additive regression model using components: piecewise growth curve trend, fourier series, dummy variables
    #pred_ARCH(data, period=365) # Autoregressive Conditional Heteroskedasticity
    #"""
    
    # risk
    var = VaR(data) # historical VaR method
    
    for k, v in var.items():
        print(f"{k}: {v}%")
    
    
    # sizing - kelly criterion
    # VaR - monte carlo
    
    
    
    
    #TODO main:
    # implied and historical volatility - to be used for VaR
    # add summary statistics: YTD yield, max yield, 1Y yield, YTD high and low
    
    
if __name__ == "__main__":
    main()