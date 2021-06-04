# portfolio_management

The main program is located in: portfolio_main_program.py

necessary functions: optimal_portfolio.py , return_portfolios.py

A python program to generate various financial models and statistics to aid the investor in the decision making process. The program will continue to be built upon.

With tools and knowledge I learned from the Codecademy courses I took as well as various online research, I created a Python program in PyCharm that will compute and visualize a Markowitz efficient frontier based on 6 user-entered tickers. The program also asks for the user's desired monthly rate of return, however many of the computations and visualizations currently represent the daily return. However, the corner portfolios closest to the user-entered monthly desired rate of return are also highlighted and selected for the user's reference. The data is pulled from the Yahoo Finance API via the pandas-datareader library. The weights, return, and risk of each optimal portfolio are also placed into a .csv file for the user's convenience and reference. The optimal_portfolio and return_portfolios functions are from my Codecademy course. However I made some small changes I felt were applicable. Many similar functions can be found with a quick internet search. The date range is currently from 2016-01-01 to the present day the program is run.
Various statistics and graphical representations of the data are also provided in the program. For example, a Jarque-Bera normality test, Covariance Matrix, daily returns of a select ticker, overall expected returns for each corner portfolio, and the distribution of returns per each entered ticker in the portfolio. Samples of these visualizations can be found in main.
