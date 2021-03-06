# Mean_Variance_Portfolio_Optimizer
Mean-Variance Portfolio Optimizer Program

This program is written in Python. It will create a portfolio for the stocks you decide to include, based on a mean-variance framework using 5 years of historical data from Tiingo. 
It will show you 5 things:
1) The maximum sharpe-ratio portfolio, the weights of the stocks in the portfolio, and it's expected risk & return.
2) The minumum variance portfolio, the weights of the stocks in the portfolio, and it's expected risk & return.
3) A Bar graph showing the risk and return tradeoff for the stocks in your portfolio.
4) The efficient frontier created with 5000 randomly-weighted portfolios, and where the maximum sharpe portfolio lies.
5) Optionally, the exact amount of each stock to buy constrained by a user-inputted portfolio value

You will need the following packages:
matplotlib
numpy
pyportfolioopt
datetime

You will need a free API key from Tiingo.
