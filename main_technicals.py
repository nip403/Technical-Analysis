from toolkit.utils import write, process_data, process_data_ticker
from toolkit.core import exit_strategy
from toolkit.technicals import *
import os
    
def main():
    path = os.path.dirname(__file__) + "\\"
    stock_data = path + "STOCK_DATA.xlsx"
    
    data = process_data(stock_data, "COUR")
    #data = data.tail(1000) # last ~2 yrs of trading
    write(stock_data, data)
    
    # calculate position sizing using 2% rule
    print(exit_strategy(
        stop_loss=0,
        price=6.6,
        pct_portfolio=None,
        pct_capital_to_risk=0.02,
    ))
    """
    # full dashboard of all indicators
    all_visual(data)
    
    # historical
    stock(data)
    SMA(data) # momentum/trend - SMA 50 & 200, MACD
    RSI(data) # momentum/trend - RSI, stochastic RSI, %K, %D
    FI(data) # volume - force index
    BB(data) # volatility - bollinger bands low mid high
    Fib(data, extension=False) # momentum/trend - fib retracement + extension
    
    # future
    pred_prophet(data, period=365) # facebook's prophet model - additive regression model using components: piecewise growth curve trend, fourier series, dummy variables
    pred_ARCH(data, period=365) # Autoregressive Conditional Heteroskedasticity
    """
    # individual risk
    var, es = PortfolioToolkit.risk(data) # historical VaR method
    
    # portfolio analysis
    symbols = "IAU,USRT,LOUP,ICLN,COUR,CD".split(",")
    weights = [32, 15, 14, 11, 2, 26]
    
    port = PortfolioToolkit(
        100_000, 
        symbols,
        weights,
        path=stock_data
    )
    
    port.plot_allocation()
    port.portfolio_returns()
    risk = port.historical_risk() # var/es for each component and total portfolio, expressed in % of total
   
    print(risk)
   
    var, es = port.monte_carlo( # var/es calculated using monte carlo simulation, 100k iterations, 10 day/99% conf int default
        simulations=10_000,
        show=True, # plot predicted cumulative daily return over simulated period & histogram of overall return distribution
    ) 
    
    
    
if __name__ == "__main__":
    main()