import numpy as np
import pandas as pd
from simplebooster import getMyPosition as getPosition


nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))


def simPrices(prcHist):
    n = 252
    t = 1
    tau = t / n
    prcHist = pd.DataFrame(prcHist.T)
    returns = prcHist.pct_change()

    sigma = returns.iloc[:-500, :].std()

    mean = returns.mean()
    mu = mean - (0.5 * sigma**2)
    S = prcHist.iloc[-1, :]
    simmedPrices = np.zeros((n, len(prcHist.columns)))
    simmedPrices[0, :] = S

    for i in range(1, n):
        Z = np.random.normal(0, 1)
        # Simulate log prices
        simmedPrices[i, :] = (
            simmedPrices[i - 1, :]
            + (mu * simmedPrices[i - 1, :])
            + (sigma * simmedPrices[i - 1, :] * Z)
        )

    # Convert simulated log prices back to normal prices
    simmedPrices = pd.DataFrame(simmedPrices)

    return simmedPrices


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(1001, 1251):
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume, todayPLL)

simulated_prices = simPrices(prcAll).T
prcFull = pd.concat([pd.DataFrame(prcAll), pd.DataFrame(simulated_prices)], axis=1, ignore_index=True).values

(meanpl, ret, plstd, sharpe, dvol, todayPLL) = calcPL(prcFull)
score = meanpl - 0.1 * plstd
print("=====")
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)
