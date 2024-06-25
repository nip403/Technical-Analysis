from core import fibonacci, _prophet, _ARCH
from utils import plotter
import pandas as pd
import numpy as np
import random

"""
Kwargs:

candlestick: bool
volume: bool
"""

# TODO: parabolic sar + aroon

# altered ta module source wrapper.py
# https://github.com/bukosabino/ta

@plotter(plot_candlestick=True, plot_volume=True)
def stock(data: pd.DataFrame, **kwargs) -> None:
    ax = kwargs.get("ax1")
    ax.set_title("Stock Price & Volume")

@plotter(False)
def SMA(data: pd.DataFrame, **kwargs) -> None: # 50/200-day SMA, MACD (100-25 windows) 
    # TODO maybe somehow add an ichimoku cloud
    ax1 = kwargs.get("ax1")
    ax2 = kwargs.get("ax2")
    
    ax1.set_title("50-day & 200-day SMA")

    ax1.plot(data["Date"], data["trend_sma_slow"], label="SMA 200", color="red")
    ax1.plot(data["Date"], data["trend_sma_fast"], label="SMA 50", color="blue")
    
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.plot(data["Date"], data["trend_macd"], label="MACD", color="green")
    ax2.axhline(data.iloc[-1]["trend_macd"], color="grey", linestyle="--")
    
@plotter(False)
def BB(data: pd.DataFrame, **kwargs) -> None: # 200 period bollinger bands
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
def RSI(data: pd.DataFrame, **kwargs) -> None: # RSI, Stochastic RSI, %K, %D - 50 day window
    ax1, ax2, ax3, ax4, ax5 = [kwargs.get(f"ax{p+1}") for p in range(5)]
    ax1.set_title("RSI")
    
    ax2.plot(data["Date"], data["momentum_rsi"], label="RSI", color="black")
    ax3.plot(data["Date"], data["momentum_stoch_rsi"], label="Stochastic RSI", color="black")
    ax4.plot(data["Date"], data["momentum_stoch_rsi_k"], label="%K", color="black")
    ax5.plot(data["Date"], data["momentum_stoch_rsi_d"], label="%D", color="black")
    
@plotter(False, rows=2, ratios=[5, 2], plot_volume=True)
def FI(data: pd.DataFrame, **kwargs) -> None: # period 25, consider adding EMV oscillator but test data looks off
    ax1 = kwargs.get("ax1")
    ax2 = kwargs.get("ax2")
    
    ax1.set_title("Force Index") 
    ax2.plot(data["Date"], data["volume_fi"], label="Force Index", color="black")
    ax2.axhline(0, color="gray", linestyle="--")
    
@plotter()
def Fib(data: pd.DataFrame, **kwargs) -> None:
    ax = kwargs.get("ax1")
    
    levels = fibonacci(data.tail(data.shape[0] // 2)) # only takes latest half of data
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2, 2.618]
    
    for i, lvl in enumerate(levels[1:]):
        ax.axhspan(
            levels[i], 
            levels[i+1], 
            color=f"#{''.join(f'{random.randint(0, 255):02X}' for _ in range(3))}", 
            alpha=0.2, 
            label=f"{round(ratios[i + 1] * 100, 1):>6}% Level: {round(lvl, 2)}"
        )

@plotter(rescale=[0])
def pred_prophet(data: pd.DataFrame, *, period, **kwargs) -> None:  
    ax = kwargs.get("ax1")  
    model, forecast = _prophet(data, period)
    ax.set_xlim(data["Date"].iloc[0], forecast["ds"].iloc[-1])
    model.plot(forecast, ax=ax)
    
@plotter(rescale=[0])
def pred_ARCH(data: pd.DataFrame, *, period, **kwargs) -> None:  
    ax = kwargs.get("ax1")  
    forecast = _ARCH(data, period)
    ax.set_xlim(data["Date"].iloc[0], forecast.index[-1])
    ax.plot(forecast.index, forecast["forecast"], label="ARCH Prediction", color="red")
