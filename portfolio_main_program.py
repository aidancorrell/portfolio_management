# import necessary libraries

import pandas_datareader as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import optimal_portfolio as opt_func
import return_portfolios as ret_port_func
import scipy.stats as stats
import seaborn as sns
import cvxopt as opt
from cvxopt import blas, solvers


# choose plot style
plt.style.use('seaborn')
sns.set_palette("pastel")


# ignore certain mathematical errors (in case)
#np.seterr(divide='ignore', invalid='ignore')


# get user input(6 tickers and monthly expected return) and print
ticker1 = input("Input your first selected ticker symbol")
ticker2 = input("Input your second selected ticker symbol")
ticker3 = input("Input your third selected ticker symbol")
ticker4 = input("Input your fourth selected ticker symbol")
ticker5 = input("Input your fifth selected ticker symbol")
ticker6 = input("Input your sixth selected ticker symbol")

er_input_monthly = input("What is your monthly expected rate of return? (%)")
user_er_daily = ((float(er_input_monthly)/100) * 12) / 252
print('')
print("Desired Monthly Return: " + str(er_input_monthly) + "%")
print('')
print("Desired Daily Return: " + str(user_er_daily) + "%")


# putting user-entered tickers in a list and printing
symbols = [ticker1, ticker2, ticker3,ticker4,ticker5,ticker6]
print('')
print(symbols)

# creating numpy array out of symbols list in case of future need
symbols_array = np.array(symbols)

#backup list of selected symbols
#symbols = ['MSFT', 'AMZN', 'AAPL', 'GOOG', 'FB', 'AMD']


#Creating dates
start_date ='2016-01-01'
end_date = date.today()

# backup end date
#end_date ='2021-05-01'


#Retreiving data from yahoo finance API through pandas data-reader

stock_data = (web.get_data_yahoo(symbols, start_date, end_date)).dropna()
stock_data = pd.DataFrame(stock_data)



#Viewing data
#print(stock_data)

# slicing data pulled from yahoo to select only the adj close of each ticker
selected = list((stock_data.columns[0:len(symbols)]))


# calculating the daily returns, dropping rows with empty values

returns_daily = stock_data[selected].pct_change().dropna()
# calculating the expected return by calculating the mean
expected_returns = returns_daily.mean()
# calculating the covariance matrix of the portfolio to determine correlation between tickers
cov_daily = returns_daily.cov()
# printing covariance matrix
print(expected_returns)
print('')
print("Covariance Matrix: ")
print('')
print(cov_daily)
print('')

returnsdf = pd.DataFrame(returns_daily)
returnsdf.columns = symbols
returnsdf = returnsdf.reset_index()
#returnsdf.to_csv('daily_returns.csv')

# Gathering index and risk-free rate data for portfolio comparison and Sharpe measure computation

index_data = (web.get_data_yahoo('^GSPC', start_date, end_date)).dropna()
index_data = pd.DataFrame(index_data)
index_daily_returns = index_data['Adj Close'].pct_change().dropna()
index_expected_returns = index_daily_returns.mean()

rf_data = (web.get_data_yahoo('^TNX', start_date, end_date)).dropna()
rf_data = pd.DataFrame(rf_data)
rf_daily = rf_data['Adj Close'].pct_change().dropna()
rf = rf_daily.mean()

#print(index_expected_returns)

##########################

# calculating the standard deviation of each ticker in the portfolio
single_asset_std = np.sqrt(np.diagonal(cov_daily))
print('')

# calculating the normality of the returns of the entire portfolio
print(stats.jarque_bera(returns_daily))
print('')

# calling function to create random portfolios with corresponding risk and return relationships
df_port = ret_port_func.return_portfolios(expected_returns, cov_daily)

# calling function to produce the efficient frontier, returning weights, returns, risks,
# and the set of portfolios
weights, returns, risks, portfolios = opt_func.optimal_portfolio(returns_daily[1:])

returns_monthly = [(i*252)/12 for i in returns]

# converting portfolios set into a pandas dataframe, adding ticker symbols as column headers for readability
portfoliosdf = pd.DataFrame(portfolios)
portfoliosdf.columns = symbols
portfoliosdf = portfoliosdf.reset_index()

# appending Expected Return and Risk columns to the optimal portfolios dataframe
portfoliosdf['Expected Return'] = returns
portfoliosdf['Risk'] = risks

# calculating the MVP
mvp = portfoliosdf.iloc[99] # minimum variance portfolio(least risky)

# calculating the MRP
mrp = portfoliosdf.iloc[0] # maximum risk portfolio(most risky)

# locating the corner portfolios with the closest expected return to the user desired expected return
chosen_portfolios = portfoliosdf.iloc[(portfoliosdf['Expected Return']-user_er_daily).abs().argsort()[:2]]
chosen_portfolios = chosen_portfolios.reset_index()

# creating a new dataset to compare the chosen corner portfolios with a market index
index_chosen_port = chosen_portfolios.copy(deep=True)
index_chosen_port.loc[len(index_chosen_port.index)] = ['101', 'index', 0, 0,0,0,0,0, index_expected_returns, 1/252]
index_chosen_port['Names'] = ['Optimal Portfolio 1', 'Optimal Portfolio 2', 'S&P 500 Index']
index_chosen_port['Expected Returns Monthly'] = (index_chosen_port['Expected Return']*252)/12
index_chosen_port['Sharpe Ratio'] = ((index_chosen_port['Expected Return'] - rf) / (index_chosen_port['Risk']))*252/12
index_chosen_port.to_csv('index_chosen_port.csv')
print(chosen_portfolios)
print(index_chosen_port.head())

### work in progress ###

"""
ticker1_weight = [chosen_portfolios.iloc[0][ticker1]]
ticker2_weight = [chosen_portfolios.iloc[0][ticker2]]
ticker3_weight = [chosen_portfolios.iloc[0][ticker3]]
ticker4_weight = [chosen_portfolios.iloc[0][ticker4]]
ticker5_weight = [chosen_portfolios.iloc[0][ticker5]]
ticker6_weight = [chosen_portfolios.iloc[0][ticker6]]
#print(ticker1_weight)

chosen_returns = pd.DataFrame()
chosen_returns['Date'] = returnsdf['Date']
chosen_returns['optimal1'] = returnsdf[ticker1] * ticker1_weight
chosen_returns['optimal2'] = returnsdf[ticker2] * ticker2_weight
chosen_returns['optimal3'] = returnsdf[ticker3] * ticker3_weight
chosen_returns['optimal4'] = returnsdf[ticker4] * ticker4_weight
chosen_returns['optimal5'] = returnsdf[ticker5] * ticker5_weight
chosen_returns['optimal6'] = returnsdf[ticker6] * ticker6_weight
chosen_returns['daily_return'] = chosen_returns['optimal1'] + chosen_returns['optimal2'] + chosen_returns['optimal3'] + chosen_returns['optimal4'] + chosen_returns['optimal5'] + chosen_returns['optimal6']

#chosen_returns.to_csv('chosen_returns.csv')
"""
################

# printing optimal portfolios dataframe to a CSV file for reference
portfoliosdf.to_csv('markowitz_portfolios.csv')
#print(portfoliosdf)


# plotting efficient frontier with highlighted chosen portfolios closest to user's desired return
df_port.plot.scatter(x='Volatility', y='Returns', fontsize=12, color = 'steelblue', alpha=0.5)
plt.plot(risks, returns, color = 'mediumseagreen', marker = 'o', alpha = 0.80)
plt.plot(chosen_portfolios['Risk'], chosen_portfolios['Expected Return'], color = 'tomato', marker = '*',markersize=14)
plt.ylabel('Expected Returns(Daily)',fontsize=14)
plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
plt.title('Efficient Frontier', fontsize=24)

plt.show()

#### WORK IN PROGRESS #### visualize optimal portfolios vs index
plt.figure(figsize=(8,4))
sns.barplot(data=index_chosen_port, x='Names', y='Expected Returns Monthly', palette=['limegreen', 'turquoise', 'mediumpurple'], alpha=0.85)
plt.title('Optimal Portfolios Returns compared to Market Index')
plt.ylabel('Expected Returns(Monthly)',fontsize=12)
plt.xlabel('')
plt.show()



### visualizing an individual ticker's returns with the returns of the entire efficient frontier

plt.figure(figsize=(12,8))

plt.subplot(211)
sns.barplot(data=returnsdf, x='Date', y=ticker1)
plt.title('Daily returns ' + str(ticker1), fontsize = 20)
plt.xticks([])
plt.ylabel('Returns',fontsize=14)
plt.xlabel('Time',fontsize=14)

plt.subplot(212)
sns.barplot(data=portfoliosdf, x='index', y="Expected Return")
plt.xticks([])
plt.xlabel('')
plt.title('Markowitz Portfolios Expected Returns', fontsize=20)
plt.show()

#####


# plotting the distribution of returns of each ticker in the portfolio

plt.figure(figsize= (14,8))

plt.subplot(231)
sns.distplot(returnsdf[ticker1], color='g')
plt.title('Distribution of returns: ' + str(ticker1), fontsize = 14)

plt.subplot(232)
sns.distplot(returnsdf[ticker2], color='b')
plt.title('Distribution of returns: ' + str(ticker2), fontsize = 14)

plt.subplot(233)
sns.distplot(returnsdf[ticker3], color='purple')
plt.title('Distribution of returns: ' + str(ticker3), fontsize = 14)

plt.subplot(234)
sns.distplot(returnsdf[ticker4], color='orange')
plt.title('Distribution of returns: ' + str(ticker4), fontsize = 14)

plt.subplot(235)
sns.distplot(returnsdf[ticker5], color='red')
plt.title('Distribution of returns: ' + str(ticker5), fontsize = 14)

plt.subplot(236)
sns.distplot(returnsdf[ticker6], color='yellow')
plt.title('Distribution of returns: ' + str(ticker6), fontsize = 14)

plt.subplots_adjust(hspace=0.5)


plt.show()

#plt.figure(figsize=(14, 8))
#sns.barplot(data=chosen_returns, x="Date", y="daily_return", color = 'b')
#sns.barplot(data=returnsdf, x="Date", y=ticker1, alpha=0.6, color = 'r')
#plt.title('Daily Returns of Optimal Chosen Portfolio vs ' + str(ticker1), fontsize = 14)
#plt.legend()
#plt.show()