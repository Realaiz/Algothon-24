import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from talib import WMA
from itertools import product
import lightgbm as lgb

nInst = 50  # Number of instruments (stocks)
currentPos = np.zeros(nInst)  # Initialise current positions with zeros
call_counter = 0
lgb_model = None
last_train_day = -69

def momentum_features(prcSoFar):
  data = pd.DataFrame()
  lags = [1, 5, 21, 42, 63, 126, 189, 252]
  for lag in lags:
      data[f"return_{lag}d"] = (
        prcSoFar.pct_change(lag, fill_method=None)
        .stack()
        .pipe(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
    )

  for lag in [42, 63, 126, 189, 252]:  ##momentum diff indicators
      data[f"momentum_{lag//21}"] = data[f"return_{lag}d"].sub(data.return_21d)
  data[f"momentum_3_12"] = data[f"return_252d"].sub(data.return_63d)

  for t in [1, 5, 21]:  # target returns
      data[f"r{t}_fwd"] = data.groupby(level=0)[f"return_{t}d"].shift(-t)
  return data

def log(df):
  return np.log1p(df)

def sign(df):
  return np.sign(df)

def power(df, exp):
  return df.pow(exp)

def rank(df: pd.DataFrame) -> pd.DataFrame:
  return df.rank(axis=1, pct=True)

def scale(df: pd.DataFrame) -> pd.DataFrame:
  return df.div(df.abs().sum(axis=1), axis=0)

def lagged_ts(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
  return df.shift(t)

def diff_ts(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
  return df.diff(period)

def rollingsum_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).sum()

def rollingmean_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).mean()

def rollingweightedmean_ts(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
  return df.apply(lambda x: WMA(x, timeperiod=period))

def rollingstd_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).std()

def rollingrank_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).apply(lambda x: x.rank().iloc[-1])

def rollingproduct_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).apply(np.prod)

def rollingmin_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).min()

def rollingmax_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).max()

def maxdate_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).apply(np.argmax).add(1)

def mindate_ts(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
  return df.rolling(window).apply(np.argmin).add(1)

def rollingcorr_ts(x: pd.Series, y: pd.Series, window: int = 10) -> pd.DataFrame:
  return x.rolling(window).corr(y)

def rollingcov_ts(x: pd.Series, y: pd.Series, window: int = 10) -> pd.DataFrame:
  return x.rolling(window).cov(y)

def alpha1(prices, returns): #1
  prices_copy = prices.copy()
  returns = returns.shift(1)
  prices_copy[returns < 0] = rollingstd_ts(returns, 20)
  return (rank(maxdate_ts(power(prices_copy, 2), 5)).mul(-0.5))

def alpha2(prices): #4
  return (-1* rollingrank_ts(rank(prices), 9))

def alpha3(prices): #9
  prices_diff = diff_ts(prices, 1)
  return prices_diff.where(rollingmin_ts(prices_diff, 5) > 0, prices_diff.where(rollingmax_ts(prices_diff, 5) < 0,-prices_diff))

def alpha4(prices): #10
  prices_diff = diff_ts(prices, 1)
  return prices_diff.where(rollingmin_ts(prices_diff, 4) > 0, prices_diff.where(rollingmin_ts(prices_diff, 4) > 0, -prices_diff))

def alpha5(prices): #23
  return diff_ts(prices, 2).mul(-1).where(rollingmean_ts(prices, 20) < prices, 0)

def alpha6(prices): #24
  cond = diff_ts(rollingmean_ts(prices, 100), 100) / lagged_ts(prices, 100) <= 0.05
  return prices.sub(rollingmin_ts(prices, 100)).mul(-1).where(cond, -diff_ts(prices, 3))

def alpha7(prices, returns): #29
  return rollingmin_ts(rank(rank(scale(log(rollingsum_ts(rank(rank(-rank(diff_ts((prices - 1), 5)))), 2))))), 5).add(rollingrank_ts(lagged_ts((-1*returns), 6), 5))

def alpha8(prices, returns): #34
  return rank(rank(rollingstd_ts(returns, 2).div(rollingstd_ts(returns, 5)).replace([-np.inf, np.inf], np.nan)).mul(-1).sub(rank(diff_ts(prices, 1))).add(2))

def alpha9(prices): #46
  cond = lagged_ts(diff_ts(prices, 10), 10).div(10).sub(diff_ts(prices, 10).div(10))
  alpha = pd.DataFrame(-np.ones_like(cond), index=prices.index, columns=prices.columns)
  alpha[cond.isnull()] = np.nan
  return cond.where(cond > 0.25, -1*alpha.where(cond < 0, -diff_ts(prices, 1)))

def alpha10(prices): #49
    cond = diff_ts(lagged_ts(prices, 10), 10).div(10).sub(diff_ts(prices, 10).div(10)) >= -0.1 * prices
    return -diff_ts(prices, 1).where(cond, 1) 

def alpha11(prices): #51
  cond = diff_ts(diff_ts(prices, 10), 10).div(10).sub(diff_ts(prices, 10).div(10)) >= -0.05 * prices
  return -diff_ts(prices, 1).where(cond, 1)

def prepare_features(prices):
    data = momentum_features(prices)
    returns = (data["return_1d"].unstack()).copy()

    alpha1_result = alpha1(prices, returns)
    alpha2_result = alpha2(prices)
    alpha3_result = alpha3(prices)
    alpha4_result = alpha4(prices)
    alpha5_result = alpha5(prices)
    alpha6_result = alpha6(prices)
    alpha7_result = alpha7(prices, returns)
    alpha8_result = alpha8(prices, returns)
    alpha9_result = alpha9(prices)
    alpha10_result = alpha10(prices)
    alpha11_result = alpha11(prices)

    # Combine all alphas into a single DataFrame
    all_alphas = pd.concat([
        alpha1_result, alpha2_result, alpha3_result, alpha4_result,
        alpha5_result, alpha6_result, alpha7_result, alpha8_result,
        alpha9_result, alpha10_result, alpha11_result
    ], axis=1, keys=['alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 
                     'alpha6', 'alpha7', 'alpha8', 'alpha9', 'alpha10', 'alpha11'])

    # Reshape to match the MultiIndex structure
    all_alphas_stacked = all_alphas.stack()

    # Create MultiIndex
    time_index = range(len(prices))
    stock_index = prices.columns
    multi_index = pd.MultiIndex.from_product([time_index, stock_index], names=['time', 'stock'])

    # Reindex to ensure all combinations are present
    all_alphas_multi = all_alphas_stacked.reindex(multi_index)

    # Reshape prices to match the MultiIndex structure
    prices_stacked = prices.stack().rename('price')
    prices_stacked.index.names = ['time', 'stock']

    # Reindex prices to ensure all combinations are present
    prices_multi = prices_stacked.reindex(multi_index)

    # Prepare return features
    return_features = data[
        ["return_1d", "return_5d", "return_21d", "return_42d", "return_63d", "return_126d", "return_189d", "return_252d",
         "momentum_2", "momentum_3", "momentum_6", "momentum_9", "momentum_12", "momentum_3_12",
         "r1_fwd", "r5_fwd", "r21_fwd"
        ]
    ]

    # Combine all features: alphas, prices, and return features
    all_features = pd.concat([all_alphas_multi, prices_multi, return_features], axis=1)

    # Sort the index to ensure it's in the correct order
    all_features = all_features.sort_index()

    return all_features


def getMyPosition(prcSoFar):
    global currentPos, call_counter, lgb_model, last_train_day

    call_counter += 1
    _, nt = prcSoFar.shape

    prices = pd.DataFrame(prcSoFar.T)
    prices_copy = prices.copy()

    all_features = prepare_features(prices_copy)

    best_params = {
        "train_length": 756.0,
        "test_length": 63.0,
        "learning_rate": 0.3,
        "num_leaves": 32.0,
        "feature_fraction": 0.3,
        "min_data_in_leaf": 500.0,
        "boost_rounds": 50.0,
    }
    best_params = pd.Series(best_params)

    params = dict(boosting="gbdt", objective="regression", verbose=-1)

    dates = sorted(all_features.index.get_level_values(0).unique())

    train_dates = dates[-int(best_params["train_length"]) :]

    idx = pd.IndexSlice
    data = all_features.loc[idx[train_dates, :], :]

    labels = sorted(data.filter(like="_fwd").filter(like='prices').columns)
    features = data.columns.difference(labels).tolist()
    lookahead = 1
    label = 'price'

    # Retrain the model every 69 calls
    if lgb_model is None or nt - last_train_day >= 63:
        lgb_train = lgb.Dataset(
            data=data[features], label=data[label], free_raw_data=False
        )

        train_params = [
            "learning_rate",
            "num_leaves",
            "feature_fraction",
            "min_data_in_leaf",
        ]
        params.update(best_params.loc[train_params].to_dict())
        for p in ["min_data_in_leaf", "num_leaves"]:
            params[p] = int(params[p])

        lgb_model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=int(best_params.boost_rounds),
        )
        last_train_day = nt

    latest_data = data.loc[data.index.get_level_values(0).max()]
    latest_features = latest_data[features]
    predicted_returns = lgb_model.predict(latest_features)

    position_limit = 10000  # Maximum absolute value of position per stock

    # Calculate positions based on predicted returns
    positions = np.zeros(nInst)
    latest_prices = prices_copy.iloc[-1, :]

    for i in range(nInst):
        if predicted_returns[i] > latest_prices[i]:
            positions[i] = position_limit / latest_prices[i]  # 100% long position
        elif predicted_returns[i] < latest_prices[i]:
            positions[i] = -position_limit / latest_prices[i]  # 100% short position

    # Update the current positions
    currentPos = np.array([int(x) for x in positions])
    return currentPos
