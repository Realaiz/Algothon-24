import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


def getMyPosition(prices, window_size=10):
    num_days, num_stocks = prices.shape
    currentPos = np.zeros(num_stocks)

    def create_features(prices, window_size):
        df = pd.DataFrame(prices)
        features = []

        # Past prices
        for i in range(1, window_size + 1):
            features.append(df.shift(i))

        # Rolling mean and std
        features.append(df.rolling(window=window_size).mean())
        features.append(df.rolling(window=window_size).std())

        # Daily returns
        features.append(df.pct_change())

        feature_df = pd.concat(features, axis=1)
        feature_df.columns = [f"price_lag_{i}" for i in range(1, window_size + 1)] + [
            "rolling_mean",
            "rolling_std",
            "daily_return",
        ]

        return feature_df.dropna()

    # Create a single XGBoost model to be used for all stocks
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    scaler = StandardScaler()

    for stock in range(num_stocks):
        stock_prices = prices[:, stock]

        # Create features
        features = create_features(stock_prices, window_size)
        X = features.values[:-1]  # Features up to the second-to-last day
        y = stock_prices[
            window_size + 1 :
        ]  # Target prices starting from day window_size+1

        # Scale features
        X_scaled = scaler.fit_transform(X)

        # Train the model
        model.fit(X_scaled, y)

        # Predict next day's price
        last_features = features.iloc[-1].values.reshape(1, -1)
        next_day_prediction = model.predict(scaler.transform(last_features))

        # Determine position based on prediction
        last_known_price = stock_prices[-1]
        if next_day_prediction > last_known_price:
            currentPos[stock] = 100  # Buy (you can adjust this value)
        elif next_day_prediction < last_known_price:
            currentPos[stock] = -100  # Sell (you can adjust this value)
        # If prediction equals last known price, position remains 0

    return currentPos


# Example usage:
# prices = np.random.rand(750, 50)  # Replace with your actual data
# positions = getMyPosition(prices)
# print(positions)
