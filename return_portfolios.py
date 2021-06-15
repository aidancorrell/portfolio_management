import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers


def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []

    selected = (expected_returns.axes)[0]
    selected = pd.DataFrame(selected)

    #print(selected)

    num_assets = len(selected)
    num_portfolios = 5000

    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}

    for counter, symbol in enumerate(selected):
        portfolio[str(symbol) + ' Weight'] = [Weight[counter] for Weight in stock_weights]

    df = pd.DataFrame(portfolio)

    column_order = ['Returns', 'Volatility'] + [str(stock) + ' Weight' for stock in selected]

    df = df[column_order]

    return df

