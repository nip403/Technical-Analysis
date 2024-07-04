from toolkit.utils import write, process_data, process_data_ticker
from toolkit.core import VaR, exit_strategy
from toolkit.visuals import *
import os
    
def main():
    path = os.path.dirname(__file__) + "\\"
    stock_data = path + "STOCK_DATA.xlsx"
    
    data = process_data(stock_data, "COUR")
    #data = data.tail(1000) # last ~2 yrs of trading
    write(stock_data, data)
    
    print(exit_strategy(
        stop_loss=0,
        price=6.6,
        pct_portfolio=None,
        pct_capital_to_risk=0.02,
    ))
    
    all_visual(data)
    
    # Historical
    stock(data)
    SMA(data) # momentum/trend - SMA 50 & 200, MACD
    RSI(data) # momentum/trend - RSI, stochastic RSI, %K, %D
    FI(data) # volume - force index
    BB(data) # volatility - bollinger bands low mid high
    Fib(data, extension=False) # momentum/trend - fib retracement + extension
    
    # Future
    pred_prophet(data, period=365) # facebook's prophet model - additive regression model using components: piecewise growth curve trend, fourier series, dummy variables
    pred_ARCH(data, period=365) # Autoregressive Conditional Heteroskedasticity

    # risk
    var = VaR(data) # historical VaR method
    
    for k, v in var.items():
        print(f"{k}: {v}%")
    
    
if __name__ == "__main__":
    main()