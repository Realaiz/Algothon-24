import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

nInst = 50  # Number of instruments (stocks)
currentPos = np.zeros(nInst)  # Initialise current positions with zeros
cached_models = [None] * nInst  # Initialise cache for ARIMA models
cached_predictions = [None] * nInst  # Initialize cache for predictions
last_eval_day = [
    -30
] * nInst  # Initialize the last evaluation day to ensure models are built on the first call
alpha = 0.07  # Smoothing factor for EMA, between 0 and 1
smoothed_predictions = np.zeros(nInst)  # Initialise smoothed predictions
volatility_threshold = 0.04  # Threshold for volatility-based position sizing


# Function to determine the best transformation and fit an ARIMA model
def fit_arima(stock_series):
    # Test original series
    pval_original = adfuller(stock_series)[1]
    if pval_original < 0.05:
        best_series = stock_series
    else:
        # Test differenced series
        diff_series = np.diff(stock_series)
        pval_diff = adfuller(diff_series)[1]

        # Test log-differenced series
        log_mask = stock_series > 0
        log_series = np.log(stock_series[log_mask])
        log_diff_series = np.diff(log_series)
        pval_log_diff = adfuller(log_diff_series)[1]

        # Choose the best transformation
        pvals = [pval_original, pval_diff, pval_log_diff]
        transformations = [stock_series, diff_series, log_diff_series]
        best_series = transformations[np.argmin(pvals)]

    # Fit ARIMA model on the best transformed series and get the forecast in one step
    model = ARIMA(best_series, order=(2, 1, 3))
    forecast = model.fit().forecast(steps=1)[0]
    return model, forecast


def momentum_alphas(prcSoFar):
    data = pd.DataFrame()
    lags = [30, 60, 90, 180, 270, 360]
    for lag in lags:
        data[f"return_{lag//30}m"] = (
            prcSoFar.pct_change(lag)
            .stack()
            .pipe(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
            .add(1)
            .pow(1 / (lag / 30))
            .sub(1)
        )
    data = data.swaplevel().dropna()  ##returns

    for lag in [60, 90, 180, 270, 360]:  ##momentum diff indicators
        data[f"momentum_{lag//30}"] = data[f"return_{lag//30}m"].sub(data.return_1m)
    data[f"momentum_3_12"] = data[f"return_12m"].sub(data.return_3m)

    for t in range(1, 7):  ##lagged returns
        data[f"return_1m_t-{t}"] = data.groupby(level=0).return_1m.shift(t)
    data.info()

    for t in [30, 60, 90, 180, 360]:  ##target returns
        data[f"target_{t//30}m"] = data.groupby(level=0)[f"return_{t//30}m"].shift(-t)

    return data


# Function to process each stock and predict its next price
def model_series(i, prcSoFar, t):
    stock_prices = prcSoFar[i]  # Get the prices of the i-th stock

    if (
        cached_models[i] is None or t - last_eval_day[i] >= 69
    ):  # Only re-evaluate models every 126 days
        model, forecast = fit_arima(stock_prices) # Fit the best ARIMA model and get the predicted price
        cached_models[i] = model  # Cache the model
        cached_predictions[i] = forecast  # Cache the forecast
        last_eval_day[i] = t  # Update the last evaluation day


    # Apply EMA to the predictions
    smoothed_predictions[i] = (
        alpha * cached_predictions[i] + (1 - alpha) * smoothed_predictions[i]
    )
    return smoothed_predictions[i]  # Return the smoothed prediction


# Main function to get the current position based on price predictions
def getMyPosition(prcSoFar):
    global currentPos  # Use the global current positions
    nInst, nt = prcSoFar.shape  # Get the number of instruments and time periods
    prcSoFar = prcSoFar[:, :-504]
    if nt < 2:
        return np.zeros(nInst)  # If not enough data points, return zero positions

    predictedPrices = [
        model_series(i, prcSoFar, nt) for i in range(nInst)
    ]  # Get the predicted prices for each stock
    
    predictedPrices = np.array(
        predictedPrices
    )  # Convert list of predicted prices to a numpy array
    latest_price = prcSoFar[:, -1]  # Get the latest prices of all stocks
    priceChanges = predictedPrices - latest_price  # Calculate the price changes

    position_limit = 10000  # Maximum absolute value of position per stock

    # Calculate the positions based on price change signal
    rpos = np.zeros(nInst)
    for i in range(nInst):
        volatility = np.std(prcSoFar[i, -30:]) / np.mean(
            prcSoFar[i, -30:]
        )  # 30-day historical volatility

        if priceChanges[i] > 0 and volatility < volatility_threshold:
            rpos[i] = position_limit / latest_price[i]  # 100% long position
        elif priceChanges[i] > 0 and volatility > volatility_threshold:
            rpos[i] = (position_limit / latest_price[i]) * 0.75  # 75% long position
        elif priceChanges[i] < 0 and volatility < volatility_threshold:
            rpos[i] = -position_limit / latest_price[i]  # 100% short position
        elif priceChanges[i] < 0 and volatility > volatility_threshold:
            rpos[i] = -(position_limit / latest_price[i]) * 0.75 # 75% short position

    # Update the current positions
    currentPos = np.array([int(x) for x in rpos])

    return currentPos  # Return the updated positions   
