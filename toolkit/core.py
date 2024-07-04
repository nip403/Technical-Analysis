import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import statsmodels.api as sm
from arch.unitroot import KPSS, ADF
import itertools as it
import warnings
import sys
from .utils import norm, _to_pct

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

def fibonacci(df: pd.DataFrame, ext: bool) -> pd.DataFrame:
    highest_swing, lowest_swing = -1, -1
    
    df.index = range(df.shape[0])
    
    for i in range(1, df.shape[0] - 1):
        if df["High"][i] > df["High"][i-1] and df["High"][i] > df["High"][i+1] and (highest_swing == -1 or df["High"][i] > df["High"][highest_swing]):
            highest_swing = i
        
        if df["Low"][i] < df["Low"][i-1] and df["Low"][i] < df["Low"][i+1] and (lowest_swing == -1 or df["Low"][i] < df["Low"][lowest_swing]):
            lowest_swing = i

    max_lvl = df["High"][highest_swing]
    min_lvl = df["Low"][lowest_swing]
    
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    levels = []
    
    for r in ratios:
        if highest_swing > lowest_swing:
            levels.append(max_lvl - ((max_lvl - min_lvl) * r))
        else:
            levels.append(min_lvl + ((max_lvl - min_lvl) * r))
    
    return _extensions(sorted(levels)) if ext else sorted(levels)

def exit_strategy(*, stop_loss: float = None, price: float = None, pct_portfolio: float = None, pct_capital_to_risk: int = 0.02) -> float:
    # % portfolio as a decimal e.g. 0.05
    # uses 2% rule by default
    # validation to come in a future DLC
    
    if stop_loss is None: # working backwards, not recommended
        return price * (1 - (pct_capital_to_risk / pct_portfolio))
    
    if pct_portfolio is None:
        return pct_capital_to_risk / (1 - (stop_loss / price))
    
    if pct_capital_to_risk is None: # calc how much the trade risks
        return pct_portfolio * (1 - (stop_loss / price))
        
    raise

def _extensions(levels: list) -> list:   
    extensions = [1, 1.618, 2, 2.618]
    
    for i in range(len(extensions) - 1): # cursed but it works
        levels.append(extensions[i + 1] / extensions[i] * levels[-1])
        
    return levels

def _prophet(data: pd.DataFrame, period: int) -> Prophet: # https://github.com/facebook/prophet
    data = data.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]
    model = Prophet()
    
    model.fit(data)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    return model, forecast

def _ARCH(data: pd.DataFrame, period: int) -> pd.DataFrame: # very expensive to reconfigure
    latest = data.iloc[-1]["Date"]
    futuredates = pd.date_range(
        start=latest + pd.DateOffset(days=1),
        periods=period,
    )
    
    data = data.set_index("Date").rename(columns={"Close": "Price"})[["Price"]]
    data["log_rtn"] = np.log(data.Price / data.Price.shift(1))
    data = data.dropna(axis=0)

    """
    # Model Selection
    results = []
    d, D = 1, 1
    
    for param in it.product(*(range(3) for _ in range(4))):
        try:
            model = sm.tsa.statespace.SARIMAX(
                data.Price, 
                order=(param[0], d, param[1]), 
                seasonal_order=(param[2], D, param[3], 12)
            ).fit(disp=-1)
        except ValueError:
            print('wrong parameters:', param)
            continue
        
        #print(param)
        results.append([param, model.bic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ["parameters", "bic"]
    
    # best model
    bic_model=sm.tsa.statespace.SARIMAX(
        data.Price, 
        order=(3, 1, 0), 
        seasonal_order=(0, 1, 0, 12), # 365 period way too expensive
    ).fit(disp=-1)
    """
    print("Loading SARIMAX model((3, 1, 0), (0, 1, 0, 75))")
    bic_model=sm.tsa.statespace.SARIMAX(
        data.Price, 
        order=(3, 1, 0), 
        seasonal_order=(0, 1, 0, 75), # 365 period way too expensive
    ).fit(disp=-1)
    
    #print(bic_model.summary())
    
    forecast = pd.DataFrame(
        data=np.nan,
        index=futuredates,
        columns=["forecast"],
    )
    
    data = pd.concat([data, forecast])
    data["forecast"] = bic_model.predict()
    
    new = pd.Series(bic_model.get_forecast(steps=period).predicted_mean.values, index=futuredates)
    data.loc[new.index, "forecast"] = new.array
    
    return data

def _exp_shortfall(returns: pd.Series, conf: float = 0.99, period: int = 1) -> float:
    return returns[returns <= np.percentile(returns, 100 * (1 - conf))].mean() * np.sqrt(period)
    
def _var(returns: pd.Series, conf: float = 0.99, period: int = 1) -> float:
    return np.percentile(returns, 100 * (1 - conf)) * np.sqrt(period)

def VaR(prices: pd.DataFrame, conf: float = 0.99, period = 10, _close_col: str = "Close", _portfolio: bool = False) -> float:    
    # conf: 0.99 = 99% confidence level
    # period: 10 day holding period
    
    trading_periods = 250 
    returns = prices.set_index("Date")[_close_col].sort_index().head(trading_periods + 1).pct_change().dropna() # off by 1 as d/dt(close) = returns
    
    assert returns.shape[0] # array of close prices cannot be all consecutive 0s when representing a stable asset
    
    var = _var(returns, conf, period)
    es = _exp_shortfall(returns, conf, period)
    
    return _to_pct({ # % of $ investment
        r"99% VaR (10-day)": var,
        r"99% Expected Shortfall (10-day)": es,
    }) if not _portfolio else ( # terrible readability, todo
        [_var(returns, conf, period), _exp_shortfall(returns, conf, period)], returns
    )

def risk_portfolio(prices: dict[str, pd.DataFrame | pd.Series], weights: dict[str, float], conf: float = 0.99, period: int = 10, _close_col: str = "Close"):
    """Calculate VaR/ES for each stock/instument and for the overall portfolio

    ALL VALUES IN PERCENTAGES

    Args: (according to FRB regulation)
        prices (pd.DataFrame): dict of pd dataframes/series for closing prices (length >= 251): {ticker: df, ticker2: df2}, requires "Date" and "Close" as columns, pass in adjusted close header if available
        weights (dict): dict of weights for each ticker in prices df, ratio is fine (will be normalised)
        conf (float, optional): confidence interval 0-1. Defaults to 0.99.
        period (int, optional): time period. Defaults to 10.
        _close_col (str, optional): pd column name to calculate from.
    """
    
    out = {}
    portfolio = {}

    assert set(weights.keys()) == set(prices.keys())
    assert sum(weights.values()) > 0
    
    # normalise weights
    weights = dict(zip(weights.keys(), norm(weights.values())))
    
    # var/es for each component
    for symbol, data in prices.items():
        out[symbol], returns = VaR(data, conf, period, _close_col, _portfolio=True)
        portfolio[symbol] = returns.reset_index(drop=True) * weights[symbol]
    
    port = pd.concat(portfolio.values(), axis=1, keys=portfolio.keys()).sum(axis=1)
    out["Total"] = [
        _var(port, conf, period), 
        _exp_shortfall(port, conf, period)
    ]
    
    return out

def aggregate_portfolio(portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ Intended usage through visuals.portfolio_returns() 
    
    Compatible directly with output from utils.process_data_multiple() 
    Input e.g.: {
        ticker: pd.DataFrame[["Date", "Close"]],
        ...
    }
    
    Returns: df of close prices
    """
    
    for name, data in portfolio.items():
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)
        portfolio[name] = data[["Close"]]
        
    new = pd.concat(list(portfolio.values()), axis=1, join="inner").dropna()
    new.columns = list(portfolio.keys())
    
    return new

def spot_portfolio_returns(portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ Intended usage through visuals.portfolio_returns() 
    
    Compatible directly with output from utils.process_data_multiple() 
    Input e.g.: {
        ticker: pd.DataFrame[["Date", "Close"]],
        ...
    }
    
    Returns: df of returns day to day
    """
    print(portfolio)
    return aggregate_portfolio(portfolio).pct_change().dropna()

def cum_portfolio_returns(portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ Intended usage through visuals.portfolio_returns() 
    
    Compatible directly with output from utils.process_data_multiple() 
    Input e.g.: {
        ticker: pd.DataFrame[["Date", "Close"]],
        ...
    }
    
    Returns: df of cumulative returns
    """
    print(portfolio)    
    return (1 + aggregate_portfolio(portfolio).pct_change()).cumprod().dropna()