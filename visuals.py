from core import fibonacci, _prophet, _ARCH
from utils import plotter, rand_hex, norm
from utils import AxisHandler as Ax
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Kwargs:

candlestick: bool
volume: bool
"""

# TODO: parabolic sar + aroon

# altered ta module source wrapper.py
# https://github.com/bukosabino/ta

figsize = (15, 9)

@plotter(plot_candlestick=True, plot_volume=True)
def stock(data: pd.DataFrame, **kwargs) -> None:
    ax = kwargs.get("ax1")
    ax.set_title("Stock Price & Volume")

@plotter(False)
def SMA(data: pd.DataFrame, **kwargs) -> None: # 50/200-day SMA, MACD (100-25 windows) 
    # TODO maybe somehow add an ichimoku cloud
    ax = Ax(kwargs)
    
    ax[1].set_title("50-day & 200-day SMA")

    ax[1].plot(data["Date"], data["trend_sma_slow"], label="SMA 200", color="red")
    ax[1].plot(data["Date"], data["trend_sma_fast"], label="SMA 50", color="blue")
    
    ax[2].axhline(0, color="gray", linestyle="--")
    ax[2].plot(data["Date"], data["trend_macd"], label="MACD", color="green")
    ax[2].axhline(data.iloc[-1]["trend_macd"], color="grey", linestyle="--")
    
@plotter(False, rows=3, ratios=[4, 1, 1])
def BB(data: pd.DataFrame, **kwargs) -> None: # 200 period bollinger bands
    ax = Ax(kwargs)
    
    ax[1].set_title("Bollinger Bands")
    
    ax[1].plot(data["Date"], data["volatility_bbh"], label="High Band", color="green")
    ax[1].plot(data["Date"], data["volatility_bbm"], label="Middle Band", color="blue")
    ax[1].plot(data["Date"], data["volatility_bbl"], label="Low Band", color="red")
    
    ax[2].plot(data["Date"], data["volatility_bbp"], label="%B", color="red")
    ax[2].axhline(1, color="gray", linestyle="--")
    ax[2].axhline(0, color="gray", linestyle="--")  
    
    ax[3].plot(data["Date"], data["volatility_bbw"], label="BBW", color="purple")

@plotter(False, rows=5, ratios=[10, 2, 2, 1, 1], plot_candlestick=True)
def RSI(data: pd.DataFrame, **kwargs) -> None: # RSI, Stochastic RSI, %K, %D - 50 day window
    ax = Ax(kwargs)
    ax[1].set_title("RSI")
    
    ax[2].plot(data["Date"], data["momentum_rsi"], label="RSI", color="black")
    ax[3].plot(data["Date"], data["momentum_stoch_rsi"], label="Stochastic RSI", color="black")
    ax[4].plot(data["Date"], data["momentum_stoch_rsi_k"], label="%K", color="black")
    ax[5].plot(data["Date"], data["momentum_stoch_rsi_d"], label="%D", color="black")
    
@plotter(False, rows=2, ratios=[5, 2], plot_volume=True)
def FI(data: pd.DataFrame, **kwargs) -> None: # period 25, consider adding EMV oscillator but test data looks off
    ax1 = kwargs.get("ax1")
    ax2 = kwargs.get("ax2")
    
    ax1.set_title("Force Index") 
    ax2.plot(data["Date"], data["volume_fi"], label="Force Index", color="black")
    ax2.axhline(0, color="gray", linestyle="--")
    
@plotter()
def Fib(data: pd.DataFrame, extension: bool = True, **kwargs) -> None:
    ax = kwargs.get("ax1")
    
    levels = fibonacci(data.tail(data.shape[0] // 2), extension) # only takes latest half of data
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2, 2.618]
    
    for i, lvl in enumerate(levels[1:]):
        if not i:
            ax.axhspan(
                levels[i - 1],
                levels[i],
                color="none",
                alpha=0,
                label=f"{round(ratios[i] * 100, 1):>6}% Level: {round(levels[i], 2)}"
            )
        
        ax.axhspan(
            levels[i], 
            levels[i + 1], 
            color=rand_hex(), 
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

def all_visual(data: pd.DataFrame, **kwargs):
    kwargs["figsize"] = (10, 16)
    kwargs["DEBUG_1"] = True
    _all_visual(data)
    
@plotter(False, rows=10, ratios=[4, 4, 2, 4, 2, 2, 1, 1, 1, 8], plot_candlestick=True, plot_volume=True)
def _all_visual(data: pd.DataFrame, **kwargs) -> None:
    ax = Ax(kwargs)
    
    # main: 2
    # SMA: 2
    ax[2].plot(data["Date"], data["Close"], label="Price", color="black")
    ax[2].plot(data["Date"], data["trend_sma_slow"], label="SMA 200", color="red")
    ax[2].plot(data["Date"], data["trend_sma_fast"], label="SMA 50", color="blue")
    
    # MACD: 1
    ax[3].axhline(0, color="gray", linestyle="--")
    ax[3].plot(data["Date"], data["trend_macd"], label="MACD", color="green")
    ax[3].axhline(data.iloc[-1]["trend_macd"], color="grey", linestyle="--")
    
    # BB: 2
    ax[4].plot(data["Date"], data["Close"], label="Price", color="black")
    ax[4].plot(data["Date"], data["volatility_bbh"], label="High Band", color="green")
    ax[4].plot(data["Date"], data["volatility_bbm"], label="Middle Band", color="blue")
    ax[4].plot(data["Date"], data["volatility_bbl"], label="Low Band", color="red")
    
    # %B: 1
    ax[5].plot(data["Date"], data["volatility_bbp"], label="%B", color="red")
    ax[5].axhline(1, color="gray", linestyle="--")
    ax[5].axhline(0, color="gray", linestyle="--")  
    
    # BBW: 1
    ax[6].plot(data["Date"], data["volatility_bbw"], label="BBW", color="purple")
    
    # RSI stuff + FI: 0.5 each
    ax[7].plot(data["Date"], data["momentum_rsi"], label="RSI", color="black")
    ax[8].plot(data["Date"], data["momentum_stoch_rsi"], label="Stochastic RSI", color="black")
    ax[9].plot(data["Date"], data["volume_fi"], label="Force Index", color="black")
    ax[10].axhline(0, color="gray", linestyle="--")
    
    # Fib: 4
    levels = fibonacci(data.tail(data.shape[0] // 2), False) # only takes latest half of data
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2, 2.618]
    
    ax[10].plot(data["Date"], data["Close"], label="Price", color="black")
    
    for i, lvl in enumerate(levels[1:]):
        ax[10].axhspan(
            levels[i], 
            levels[i+1], 
            color=rand_hex(), 
            alpha=0.2, 
            label=f"{round(ratios[i + 1] * 100, 1):>6}% Level: {round(lvl, 2)}"
        )
        
def allocation(portfolio: dict[str, int | float], title: str = "") -> None:
    """ args:
    portfolio (dict[str, int  |  float]): dict of portfolio allocations and their weights
    
    e.g.
    {
        "MSFT": 10,
        "AAPL": 20,
        "META": 30,
    }
    """
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))
    ax.set_title(title)
    lbl = [f"{round(v * 100, 1)}% - {k}" for k, v in zip(portfolio.keys(), norm(portfolio.values()))]
    
    wedges, texts = ax.pie(
        portfolio.values(), 
        wedgeprops = dict(width=0.5), 
        startangle = -40
    )
    
    kw = dict(
        arrowprops = dict(arrowstyle="-"),
        zorder = 0, 
        va = "center"
    )

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        kw["arrowprops"].update({"connectionstyle": f"angle,angleA=0,angleB={ang}"})
        ax.annotate(
            lbl[i],
            xy = (x, y), 
            xytext = (1.35*np.sign(x), 1.4*y),
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))], 
            **kw,
        )

    plt.show()