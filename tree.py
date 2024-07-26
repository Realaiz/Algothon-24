from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)


# Function to prepare data for training
def prepareData(prcAll, window_size=10):
    X, y = [], []
    for i in range(window_size, prcAll.shape[1]):
        X.append(prcAll[:, i - window_size : i].flatten())
        y.append(np.log(prcAll[:, i] / prcAll[:, i - 1]))
    return np.array(X), np.array(y)


# Function to check if a stock is fluctuating too much within a given window
def isFluctuating(prices, window=10, threshold=0.01):
    recent_returns = np.log(prices[-window:] / prices[-window - 1 : -1])
    if np.std(recent_returns) > threshold:
        return True
    return False


# Function to check if a stock has had an overall loss in the past window days
def hasOverallLoss(prices, window=10):
    profit_loss = prices[-1] / prices[-window]
    if profit_loss < 1:
        return True
    return False


# Function to get the trading position using the trained decision tree model
def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    np.random.seed(42)

    # Prepare data
    X, y = prepareData(prcSoFar)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Hyperparameter tuning for the decision tree
    param_grid = {
        "max_depth": [5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 5, 10],
    }

    model = DecisionTreeRegressor()
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_val, y_val)

    best_model = grid_search.best_estimator_
    best_model.fit(X, y)

    if nt < 10:
        return np.zeros(nins)

    # Prepare the feature input for prediction
    last_window = prcSoFar[:, -10:].flatten().reshape(1, -1)
    pred_ret = best_model.predict(last_window).flatten()

    lNorm = np.sqrt(pred_ret.dot(pred_ret))
    pred_ret /= lNorm
    rpos = np.array([int(x) for x in 50000 * pred_ret / prcSoFar[:, -1]])

    # Adjust positions based on fluctuation criteria
    for i in range(nins):
        if isFluctuating(prcSoFar[i], window=10) and hasOverallLoss(
            prcSoFar[i], window=10
        ):
            rpos[i] = 0

    currentPos = np.array([int(x) for x in currentPos + rpos])

    return currentPos
