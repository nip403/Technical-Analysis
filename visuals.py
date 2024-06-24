from utils import plotter
import pandas as pd

"""
Kwargs:

candlestick: bool
volume: bool
"""

@plotter(plot_candlestick=True, plot_volume=True)
def stock(data: pd.DataFrame, **kwargs) -> None:
    ax = kwargs.get("ax1")
    ax.set_title("Stock Price & Volume")

@plotter(False)
def SMA(data: pd.DataFrame, **kwargs) -> None:
    ax1 = kwargs.get("ax1")
    ax2 = kwargs.get("ax2")
    
    ax1.set_title("50-day & 200-day SMA")

    ax1.plot(data["Date"], data["trend_sma_slow"], label="SMA 200", color="red")
    ax1.plot(data["Date"], data["trend_sma_fast"], label="SMA 50", color="blue")
    
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.plot(data["Date"], data["trend_macd"], label="MACD", color="green")
    
@plotter(False)
def BB(data: pd.DataFrame, **kwargs) -> None:
    ax1 = kwargs.get("ax1")
    ax2 = kwargs.get("ax2")
    
    ax1.set_title("Bollinger Bands")
    
    ax1.plot(data["Date"], data["volatility_bbh"], label="High Band", color="green")
    ax1.plot(data["Date"], data["volatility_bbm"], label="Middle Band", color="blue")
    ax1.plot(data["Date"], data["volatility_bbl"], label="Low Band", color="red")
    
    ax2.plot(data["Date"], data["volatility_bbp"], label="%B", color="red")
    ax2.axhline(1, color="gray", linestyle="--")
    ax2.axhline(0, color="gray", linestyle="--")  

@plotter(False, rows=5, ratios=[10, 2, 2, 1, 1], plot_candlestick=True)
def RSI(data: pd.DataFrame, **kwargs) -> None:
    ax1, ax2, ax3, ax4, ax5 = [kwargs.get(f"ax{p+1}") for p in range(5)]
    ax1.set_title("RSI")
    
    ax2.plot(data["Date"], data["momentum_rsi"], label="RSI", color="black")
    ax3.plot(data["Date"], data["momentum_stoch_rsi"], label="Stochastic RSI", color="black")
    ax4.plot(data["Date"], data["momentum_stoch_rsi_k"], label="%K", color="black")
    ax5.plot(data["Date"], data["momentum_stoch_rsi_d"], label="%D", color="black")