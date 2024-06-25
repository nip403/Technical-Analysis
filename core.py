import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import statsmodels.api as sm
from arch.unitroot import KPSS, ADF
import itertools as it
import warnings

warnings.filterwarnings("ignore")

#from statsmodels.stats.outliers_influence import summary_table
#from sklearn.ensemble import RandomForestRegressor
#from skforecast.ForecasterAutoreg import ForecasterAutoreg

#import pmdarima as pm
#from pmdarima.arima import ndiffs, nsdiffs
#from pmdarima.metrics import smape
#from sklearn.metrics import mean_squared_error

def fibonacci(df: pd.DataFrame) -> pd.DataFrame:
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
    
    return _extensions(sorted(levels))

def _extensions(levels: list) -> list:   
    extensions = [1, 1.618, 2, 2.618]
    
    for i in range(len(extensions) - 1): # cursed but logical behaviour
        levels.append(extensions[i + 1] / extensions[i] * levels[-1])
        
    return levels

def _prophet(data: pd.DataFrame, period: int) -> Prophet: # https://github.com/facebook/prophet
    data = data.rename(columns={"Date": "ds", "Close": "y"})[["ds", "y"]]
    model = Prophet()
    
    model.fit(data)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    return model, forecast

def _ARCH(data: pd.DataFrame, period: int) -> pd.DataFrame: # very expensive if reconfiguring
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

    bic_model=sm.tsa.statespace.SARIMAX(
        data.Price, 
        order=(3, 1, 0), 
        seasonal_order=(0, 1, 0, 50), # 365 period way too expensive
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