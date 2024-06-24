import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm
from pmdarima.arima import ndiffs, nsdiffs
from pmdarima.metrics import smape
from sklearn.metrics import mean_squared_error

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

class Model:
    def __init__(self, train, test):
        self.ytrain = train
        self.ytest = test
        
        self.forecasts = []
        self.test_forecasts = []
        
    def train(self) -> None:
        kpss_diffs = ndiffs(self.ytrain["Close"].values, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(self.ytrain["Close"].values, alpha=0.05, test='adf', max_d=6)
        
        self.model = pm.auto_arima(
            self.ytrain["Close"].values, 
            d=max(adf_diffs, kpss_diffs), 
            seasonal=False, 
            stepwise=True,
            suppress_warnings=True, 
            error_action="ignore", 
            max_p=6,
            max_order=None, 
            #trace=True,
        )
        
    def test(self) -> None:
        print(self.model.predict())
        
        for i in self.ytest["Close"].values:    
            predicted = self.model.predict(n_periods=1).tolist()[0]
            self.test_forecasts.append(predicted)
            self.model.update(i)
        
        self.ytest["ARIMA_pred"] = self.test_forecasts
        
        
        print(self.model.predict())
        
    def __iter__(self):
        return self
    
    def __next__(self):  
        
        return self

def _ARIMA(data: pd.DataFrame) -> pd.DataFrame:
    train_len = int(data.shape[0] * 0.8)
    
    train_data = data.iloc[int(data.shape[0] * 0.1):train_len]
    test_data = data.iloc[train_len:]
    
    model = Model(train_data, test_data)
    model.train()
    model.test()
    
    return model