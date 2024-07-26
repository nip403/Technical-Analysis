import pandas as pd
import numpy as np
from prophet import Prophet
import statsmodels.api as sm
#from arch.unitroot import KPSS, ADF
from functools import partial
import warnings
from copy import deepcopy
import sys
from .utils import norm, _to_pct, process_data_multiple, process_ticker_multiple

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

class Risk:
    trading_periods = 250 
    
    @classmethod
    def _exp_shortfall(cls, returns: pd.Series, conf: float = 0.99, period: int = 1) -> float:
        return returns[returns <= np.percentile(returns, 100 * (1 - conf))].mean() * np.sqrt(period)
        
    @classmethod
    def _var(cls, returns: pd.Series, conf: float = 0.99, period: int = 1) -> float:
        return np.percentile(returns, 100 * (1 - conf)) * np.sqrt(period)

    @classmethod
    def risk(cls, prices: pd.DataFrame, conf: float = 0.99, period = 10, _close_col: str = "Close", _portfolio: bool = False) -> float:    
        # conf: 0.99 = 99% confidence level
        # period: 10 day holding period
        # calculate the var/es for an individual stock: df[DatetimeIndex<>
                
        try: 
            prices.set_index("Date", inplace=True)
        except:
            pass

        returns = prices[_close_col].sort_index().tail(cls.trading_periods + 1).pct_change().dropna() # off by 1 as d/dt(close) = returns
        
        assert returns.shape[0] # array of close prices cannot be all consecutive 0s (e.g. when representing a 0/low risk asset set price to 1s)
        
        var = cls._var(returns, conf, period)
        es = cls._exp_shortfall(returns, conf, period)
        
        return _to_pct({ # % of $ investment
            r"99% VaR (10-day)": var,
            r"99% Expected Shortfall (10-day)": es,
        }) if not _portfolio else [ # clunky overloading practice, terrible readability, todo
            cls._var(returns, conf, period), 
            cls._exp_shortfall(returns, conf, period)
        ]

class PortfolioBase(Risk):
    def __init__(self, portfolio_value: int | float, symbols: list[str], weights: list[int | float], data: dict[str, pd.DataFrame] = None, path: str = None):
        """General utils and Monte Carlo simulation for calculating VaR and CVaR (ES)

        Args:
            symbols (list[str]): list of ticker symbols
            weights (list[int  |  float]): list of weights in the same order
            data (str, dict[pd.DataFrame], optional): dict of dataframes for given symbols, indexed by date (most recent last) with a "Close" column. If None, data is taken from spreadsheet (path). will be prioritised even if path is provided.
            path (str, optional): path to spreadsheet, see data.
        """
        
        assert sum(weights) > 0       
         
        self.value = portfolio_value
        self.sym = symbols
        self.wt = norm(weights)
        
        if data is not None:
            self.close = data
        elif path is not None:
            self.close = process_data_multiple(path, self.sym, False)  
        else:
            self.close = process_ticker_multiple(self.sym, False)
        
        self.portfolio = self.aggregate_portfolio(self.close)
        
        self.ret = self.spot_returns(self.close)
        self.cum_ret = self.cum_returns(self.close)
        self.log_ret = self.log_returns(self.close)
        self.wt_ret = self.weighted_returns(self.ret, self.wt)
        
    @classmethod
    def aggregate_portfolio(cls, portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """ Intended usage through technicals.portfolio_returns()
        
        Compatible directly with output from utils.process_data_multiple() 
        Input e.g.: {
            ticker: pd.DataFrame[["Date", "Close"]],
            ...
        }
        
        Returns: df of close prices: DateIndex <ticker1, ticker2, ...>
        """

        portfolio = deepcopy(portfolio) # REALLY dumb bug related to not being able to pass (truly) by value i.e. memory of underlying dict values are passed for some reason
        
        for name, data in portfolio.items():
            data["Date"] = pd.to_datetime(data["Date"])
            data.set_index("Date", inplace=True)
                
            portfolio[name] = data[["Close"]]
            
        new = pd.concat(list(portfolio.values()), axis=1, join="inner").dropna()
        new.columns = list(portfolio.keys())
        
        return new
        
    @classmethod
    def spot_returns(cls, portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """ Intended usage through technicals.portfolio_returns()
        
        Compatible directly with output from utils.process_data_multiple() 
        Input e.g.: {
            ticker: pd.DataFrame[["Date", "Close"]],
            ...
        }
        
        Returns: df of returns day to day
        """

        return cls.aggregate_portfolio(portfolio).pct_change().dropna()

    @classmethod
    def cum_returns(cls, portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """ Intended usage through technicals.portfolio_returns()
        
        Compatible directly with output from utils.process_data_multiple() 
        Input e.g.: {
            ticker: pd.DataFrame[["Date", "Close"]],
            ...
        }
        
        Returns: df of cumulative returns
        """
    
        return (1 + cls.aggregate_portfolio(portfolio).pct_change()).cumprod().dropna()

    @classmethod
    def log_returns(cls, portfolio: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """ Intended usage through technicals.portfolio_returns()
        
        Compatible directly with output from utils.process_data_multiple() 
        Input e.g.: {
            ticker: pd.DataFrame[["Date", "Close"]],
            ...
        }
        
        Returns: df of log returns
        """
    
        return np.log(cls.aggregate_portfolio(portfolio)).diff().dropna()
    
    @classmethod
    def weighted_returns(cls, returns: pd.DataFrame, weights: list | np.ndarray) -> pd.DataFrame:
        return returns.mul(norm(weights), axis=1)
    
    def historical_risk(self, conf: float = 0.99, period: int = 10, _close_col: str = "Close") -> dict[str, list[float, float]]:
        """Calculate VaR/ES for each stock/instument and for the overall portfolio

        RETURNS ALL VALUES IN PERCENTAGES
        - to get individual component var: percentage * total port value * weight

        Args: (according to FRB regulation)
            conf (float, optional): confidence interval 0-1. Defaults to 0.99.
            period (int, optional): time period. Defaults to 10.
            _close_col (str, optional): pd column name to calculate from.
        """
        
        # var/es for each component
        risk = {
            symbol: self.risk(self.close[symbol], conf, period, _close_col, _portfolio=True) 
                for symbol in self.sym
        }    
        
        for k, v in risk.items():
            pos = self.sym.index(k)
            wt = self.wt[pos]
            for i in v[:]:
                risk[k].append(self.value * i * wt)
        
        risk["Total"] = [
            self._var(self.wt_ret.sum(axis=1), conf, period), 
            self._exp_shortfall(self.wt_ret.sum(axis=1), conf, period),
            self._var(self.wt_ret.sum(axis=1), conf, period) * self.value, 
            self._exp_shortfall(self.wt_ret.sum(axis=1), conf, period) * self.value,
        ]
         
        return risk

    def expected_returns(self) -> pd.DataFrame:
        """ daily expected return, assuming future returns are based on historical returns """
        
        return np.sum(self.log_ret.mean() * self.wt)

    def std(self) -> np.ndarray:
        """ returns standard deviation (=volatility) in % of the portfolio """
        
        return np.sqrt(self.wt.T @ self.log_ret.cov() @ self.wt) # log-scaled returns covariance matrix

    def _simulate(self, expected: float, std: float, days: int, _: None) -> float:
        return (expected * days) + (std * np.sqrt(days) * np.random.normal(0, 1)) # % of portfolio. multiply by portfolio value to get dollar gain/loss for period

    def monte_carlo(self, simulations: int = 1_000, days: int = 10, conf: float = 0.99, show: bool = True):
        """Calculate VaR and CVaR (ES) using Monte Carlo method.

        Args - all default to FRB regulation:
            simulations (int, optional): number of simulations to run
            days (int, optional): VaR period
            conf (float, optional): confidence interval
            plotter (class visuals.monte_carlo): for plotting results of simulations
            
        Returns: n-day (param) value at risk, n-day expected shortfall
        """
        std = self.std() # standard deviation of portfolio
        exp = self.expected_returns() # daily expected return (%)
        
        # monte carlo simulation - constant over period
        """ => averaged simulation over period 
        simul = partial(self._simulate, exp, std, days)
        returns = np.frompyfunc(
            simul, 1, 1
        )(np.arange(simulations))
        
        
        var = -np.percentile(returns, 100 * (1 - conf))
        es = returns[returns <= var].mean()
        """
        
        # monte carlo simulation - variable over period
        daily = np.frompyfunc(partial(self._simulate, exp, std, 1), 1, 1)(np.zeros((simulations, days)))
        returns = np.cumprod(1 + daily, axis=1)[:, -1] - 1
        var = -np.percentile(returns, 100 * (1 - conf))
        
        if hasattr(self, "_plot_monte_carlo") and show: # some serious off by 1 action to be able to show cumulative returns
            self._plot_monte_carlo(
                returns + 1,
                daily,
                days,
                conf,
                var - 1,
            )
        
        return -var, returns[returns <= -var].mean() 